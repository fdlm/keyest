from docopt import docopt
from os.path import join, exists
import os
import yaml
import tqdm
import data
import pickle
import numpy as np
import lasagne as lnn
import theano
import theano.tensor as tt
from lasagne_tools import SequenceIterator

USAGE = """
Usage:
    train.py [options]

Options:
    --n_preproc_layers=I  Number of preprocessing layers [default: 0]
    --n_preproc_units=I  Number of preprocessing units [default: 64]
    --preproc_dropout=F  Dropout probability in preprocessing [default: 0.5]
    --n_combiner_units=I  Number of combiner units [default: 24]
    --batch_size=I  Batch Size to use [default: 8]
    --no_dist_sampling  do not use distribution sampling
    --n_epochs=I  number of epochs to train [default: 1000]
    --exp_id=S  output directory [default: last_exp]
    --feature=S  feature to use (lfs/dc) [default: lfs]
"""


class AverageLayer(lnn.layers.MergeLayer):

    def __init__(self, incoming, axis, mask_input=None, **kwargs):
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        self.axis = axis
        super(AverageLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[:self.axis] + input_shape[self.axis + 1:]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        if mask is None:
            return tt.mean(input, axis=self.axis)
        else:
            mask = mask.dimshuffle(0, 1, 'x')
            return tt.sum(input * mask, axis=self.axis) / tt.sum(mask, axis=1)


def build_model(feature_size,
                n_preproc_layers,
                n_preproc_units,
                preproc_dropout,
                n_combiner_units):

    from lasagne.layers import DenseLayer, InputLayer, DropoutLayer, ReshapeLayer
    from lasagne.nonlinearities import identity, softmax, elu

    net = InputLayer((None, None, feature_size))
    mask = InputLayer((None, None))
    input_var = net.input_var
    mask_var = mask.input_var
    n_batch, n_time_steps, _ = input_var.shape
    net = ReshapeLayer(net, (-1, feature_size))
    for i in range(n_preproc_layers):
        net = DenseLayer(net, num_units=n_preproc_units, nonlinearity=elu)
        if preproc_dropout > 0.:
            net = DropoutLayer(net, p=preproc_dropout)

    net = DenseLayer(net, num_units=n_combiner_units, nonlinearity=elu)
    net = ReshapeLayer(net, (n_batch, n_time_steps, n_combiner_units))
    net = AverageLayer(net, axis=1, mask_input=mask)
    net = DenseLayer(net, num_units=24, nonlinearity=softmax)

    return net, input_var, mask_var


def main():
    args = docopt(USAGE)
    n_preproc_layers = int(args['--n_preproc_layers'])
    n_preproc_units = int(args['--n_preproc_units'])
    preproc_dropout = float(args['--preproc_dropout'])
    n_combiner_units = int(args['--n_combiner_units'])
    batch_size = int(args['--batch_size'])
    no_dist_sampling = args['--no_dist_sampling']
    n_epochs = int(args['--n_epochs'])
    exp_id = args['--exp_id']
    feature = args['--feature']

    print args

    exp_dir = join('results', exp_id)
    if not exists(exp_dir):
        os.makedirs(exp_dir)
    yaml.dump(args, open(join(exp_dir, 'config'), 'w'))

    print 'Loading GiantSteps Dataset...'

    test_dataset = data.load_giantsteps_key_dataset(
        'data/giantsteps-key-dataset-augmented',
        'feature_cache',
        feature
    )

    test_set = data.load_data(
        test_dataset.all_files(),
        use_augmented=False
    )

    print 'Loading GiantSteps MTG Dataset...'

    train_dataset = data.load_giantsteps_key_dataset(
        'data/giantsteps-mtg-key-dataset-augmented',
        'feature_cache',
        feature
    )

    training_files, val_files = train_dataset.random_split([0.8, 0.2])
    training_set = data.load_data(
        training_files,
        use_augmented=True
    )
    val_set = data.load_data(
        val_files,
        use_augmented=False
    )

    if not no_dist_sampling:
        l = [np.load(f) for f in training_files['targ'] if '.0.' in f]
        targ_dist = np.bincount(np.hstack(l), minlength=24).astype(np.float)
        targ_dist /= targ_dist.sum()
    else:
        targ_dist = None

    model, X, m = build_model(
        feature_size=training_set[0][0].shape[-1],
        n_preproc_layers=n_preproc_layers,
        n_preproc_units=n_preproc_units,
        preproc_dropout=preproc_dropout,
        n_combiner_units=n_combiner_units
    )

    y = tt.ivector('y')
    y_hat = lnn.layers.get_output(model, deterministic=False)
    loss = tt.mean(lnn.objectives.categorical_crossentropy(y_hat, y),
                   dtype='floatX')
    acc = tt.mean(lnn.objectives.categorical_accuracy(y_hat, y),
                  dtype='floatX')
    params = lnn.layers.get_all_params(model)
    updates = lnn.updates.rmsprop(loss, params, learning_rate=0.001)
    train = theano.function(
        inputs=[X, y, m],
        outputs=[loss, acc],
        updates=updates)

    y_hat_test = lnn.layers.get_output(model, deterministic=True)
    loss_test = tt.mean(lnn.objectives.categorical_crossentropy(y_hat_test, y),
                        dtype='floatX')
    acc_test = tt.mean(lnn.objectives.categorical_accuracy(y_hat_test, y),
                       dtype='floatX')
    test = theano.function(
        inputs=[X, y, m],
        outputs=[loss_test, acc_test])

    train_it = SequenceIterator(training_set, batch_size=batch_size,
                                distribution=targ_dist)
    val_it = SequenceIterator(val_set, batch_size=1, shuffle=False)
    test_it = SequenceIterator(test_set, batch_size=1, shuffle=False)

    def iterate(iterator, func):
        loss = 0.0
        acc = 0.0
        n_it = 0
        batches = tqdm.tqdm(
            iterator, total=iterator.n_elements, leave=False)
        for batch in batches:
            X_batch, y_batch, m_batch = zip(*batch)
            X_batch = np.stack(X_batch)
            l, a = func(X_batch, y_batch, m_batch)
            loss += l
            acc += a
            n_it += 1
        return loss / n_it, acc / n_it

    outputs = ['epoch', '     loss', '    accuracy',
               '    validation loss', '    validation accuracy']

    print ''.join(outputs)

    best_val_acc = -np.inf
    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss, train_acc = iterate(train_it, train)
        val_loss, val_acc = iterate(val_it, test)

        tqdm.tqdm.write('{:5d}\t{:4.6f}\t{:10.4f}\t{:13.6f}\t{:19.4f}'.format(
            epoch, float(train_loss), float(train_acc), float(val_loss),
            float(val_acc)))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            pickle.dump(
                lnn.layers.get_all_param_values(model),
                open(join(exp_dir, 'best_model'), 'wb')
            )

    lnn.layers.set_all_param_values(
        model,
        pickle.load(open(join(exp_dir, 'best_model'), 'rb'))
    )

    test_loss, test_acc = iterate(test_it, test)
    yaml.dump(
        {'loss': float(test_loss),
         'acc': float(test_acc)},
        open(join(exp_dir, 'test_results'), 'w')
    )

    print 'Test Loss: {}'.format(test_loss)
    print 'Test Accuracy: {}'.format(test_acc)


if __name__ == '__main__':
    main()

