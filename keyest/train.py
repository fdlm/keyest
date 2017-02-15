import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import data
from chainer_tools import SequenceIterator, TestModeEvaluator, ThreadedIterator
from models import DenseCombiner, BinaryTreeNet, Mlp
from augmenters import SemitoneShift, Detuning

from docopt import docopt


USAGE = """
Usage:
    train.py [options]

Options:
    --n_preproc_layers=I  Number of preprocessing layers [default: 0]
    --n_preproc_units=I  Number of preprocessing units [default: 64]
    --preproc_dropout=F  Dropout probability in preprocessing [default: 0.5]
    --combiner_type=S  Type of combiner (tree or avg) [default: tree]
    --n_combiner_units=I  Number of combiner units [default: 24]
    --combiner_dropout=F  Dropout probability in combiner [default: 0.0]
    --batch_size=I  Batch Size to use [default: 8]
    --no_dist_sampling  do not use distribution sampling
"""


class Model(chainer.Chain):

    def __init__(self, preproc, combiner, n_units):
        super(Model, self).__init__(
            preproc=preproc,
            project=L.Linear(None, n_units),
            combiner=combiner,
            output=L.Linear(n_units, 25)
        )

        self.train = True

    def set_train(self, train):
        self.preproc.set_train(train)
        self.combiner.set_train(train)
        self.train = train

    def __call__(self, x):
        batch_size = x.shape[0]
        # flatten sequences for frame-wise processing
        x = F.reshape(x, (-1,) + x.shape[2:])
        x = self.preproc(x)
        x = self.project(x)
        x = F.elu(x)
        # reshape to sequences
        x = F.reshape(x, (batch_size, -1) + x.shape[1:])
        x = self.combiner(x)
        x = self.output(x)
        return x


class Averager(chainer.Chain):

    def set_train(self, train):
        pass

    def __call__(self, x):
        return F.sum(x, axis=-2) / x.shape[-2]


def main():
    args = docopt(USAGE)
    n_preproc_layers = int(args['--n_preproc_layers'])
    n_preproc_units = int(args['--n_preproc_units'])
    preproc_dropout = float(args['--preproc_dropout'])
    combiner_type = args['--combiner_type']
    n_combiner_units = int(args['--n_combiner_units'])
    combiner_dropout = float(args['--combiner_dropout'])
    batch_size = int(args['--batch_size'])
    no_dist_sampling = args['--no_dist_sampling']

    print 'Loading GiantSteps Dataset...'

    dataset = data.load_giantsteps_key_dataset(
        'data/giantsteps-key-dataset-augmented',
        'feature_cache'
    )
    train_set, val_set, test_set, targ_dist = data.get_splits(dataset, 0, 1)
    if no_dist_sampling:
        targ_dist = None

    print 'Loading GiantSteps MTG Dataset...'
    additional_train_dataset = data.load_giantsteps_key_dataset(
        'data/giantsteps-mtg-key-dataset-augmented',
        'feature_cache'
    )
    additional_train_set = data.load_data(
        additional_train_dataset.all_files(),
        use_augmented=True
    )

    train_set += additional_train_set

    device = 0 if chainer.cuda.available else -1
    preproc = Mlp(
        n_layers=n_preproc_layers,
        n_units=n_preproc_units,
        dropout=preproc_dropout,
        activation=F.elu
    )
    if combiner_type == 'tree':
        combiner = BinaryTreeNet(
            DenseCombiner(n_combiner_units, F.elu),
            n_combiner_units,
            dropout=combiner_dropout,
            device=device
        )
    elif combiner_type == 'avg':
        combiner = Averager()
    else:
        raise ValueError('Unknown combiner type: {}'.format(combiner_type))

    model = Model(preproc, combiner, n_units=n_combiner_units)
    classifier = L.Classifier(model)

    if device == 0:
        chainer.cuda.get_device(device).use()
        classifier.to_gpu()

    opt = chainer.optimizers.RMSpropGraves()
    opt.setup(classifier)
    opt.add_hook(chainer.optimizer.GradientClipping(10.))

    augmenters = [
        # Detuning(0.5, 0.4, 2)
    ]
    train_it = SequenceIterator(train_set, batch_size=batch_size,
                                augmenters=augmenters,
                                distribution=targ_dist)
    train_it = ThreadedIterator(train_it, n_cached_items=30)
    val_it = SequenceIterator(val_set, batch_size=1,
                              repeat=False, shuffle=False)
    upd = chainer.training.updater.StandardUpdater(train_it, opt,
                                                   device=device)
    trainer = chainer.training.Trainer(upd, stop_trigger=(1000, 'epoch'))
    trainer.extend(TestModeEvaluator(val_it, classifier, device=device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration',
         'main/loss', 'main/accuracy',
         'validation/main/loss', 'validation/main/accuracy']),
        trigger=(1, 'epoch'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()


if __name__ == "__main__":
    main()
