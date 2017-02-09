import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import data
from chainer_tools import SequenceIterator, TestModeEvaluator
from models import DenseCombiner, BinaryTreeNet, Mlp
from augmenters import SemitoneShift, Detuning


class Model(chainer.Chain):

    def __init__(self, preproc, n_units, device=-1):
        super(Model, self).__init__(
            preproc=preproc,
            project=L.Linear(None, n_units),
            tree=BinaryTreeNet(
                DenseCombiner(n_units, F.elu),
                n_units,
                device=device
            ),
            output=L.Linear(n_units, 25)
        )

        self.train = True

    def set_train(self, train):
        self.preproc.set_train(train)

    def __call__(self, x):
        batch_size = x.shape[0]
        # flatten sequences for frame-wise processing
        x = F.reshape(x, (-1,) + x.shape[2:])
        x = self.preproc(x)
        x = self.project(x)
        x = F.elu(x)
        # reshape to sequences
        x = F.reshape(x, (batch_size, -1) + x.shape[1:])
        x = self.tree(x)
        x = self.output(x)
        return x


dataset = data.load_giantsteps_key_dataset('data/giantsteps-key-dataset',
                                           'feature_cache')
train_set, val_set, test_set = [data.load_data(split)
                                for split in dataset.fold_split(0, 1)]

device = 0 if chainer.cuda.available else -1
preproc = Mlp(n_layers=3, n_units=64, dropout=0.5, activation=F.elu)
model = Model(preproc, n_units=32, device=device)
classifier = L.Classifier(model)

if device == 0:
    chainer.cuda.get_device(device).use()
    classifier.to_gpu()

opt = chainer.optimizers.RMSpropGraves()
opt.setup(classifier)

augmenters = [
    # SemitoneShift(1.0, 4, 2),
    Detuning(1.0, 0.4, 2)
]
train_it = SequenceIterator(train_set, batch_size=8, augmenters=augmenters)
val_it = SequenceIterator(val_set, batch_size=1, repeat=False, shuffle=False)
upd = chainer.training.updater.StandardUpdater(train_it, opt, device=device)
trainer = chainer.training.Trainer(upd, stop_trigger=(100, 'epoch'))
trainer.extend(TestModeEvaluator(val_it, classifier, device=device))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(
    ['epoch', 'iteration',
     'main/loss', 'main/accuracy',
     'validation/main/loss', 'validation/main/accuracy']),
    trigger=(1, 'epoch'))
trainer.extend(extensions.ProgressBar(update_interval=10))
trainer.run()
