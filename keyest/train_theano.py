from __future__ import print_function
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
    train_theano.py [options]

Options:
    --n_preproc_layers=I  Number of preprocessing layers [default: 0]
    --n_preproc_units=I  Number of preprocessing units [default: 64]
    --preproc_dropout=F  Dropout probability in preprocessing [default: 0.5]
    --combiner_type=S  Type of combiner (rnn or avg) [default: avg]
    --n_combiner_units=I  Number of combiner units [default: 24]
    --batch_size=I  Batch Size to use [default: 8]
    --no_dist_sampling  do not use distribution sampling
    --n_epochs=I  number of epochs to train [default: 1000]
    --exp_id=S  output directory [default: last_exp]
    --feature=S  feature to use (lfs/dc) [default: lfs]
    --data=S  data setup (giantsteps/billboard) [default: giantsteps]
    --patience=I  number of steps to wait until lr is reduced [default: 20]
    --init_lr=F  initial learn rate [default: 0.001]
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


def build_avg_model(feature_size,
                    n_preproc_layers,
                    n_preproc_units,
                    preproc_dropout,
                    n_combiner_units):

    from lasagne.layers import (DenseLayer, InputLayer, DropoutLayer,
                                ReshapeLayer, Conv2DLayer)
    from lasagne.nonlinearities import softmax, elu

    net = InputLayer((None, None, feature_size))
    mask = InputLayer((None, None))
    input_var = net.input_var
    mask_var = mask.input_var
    n_batch, n_time_steps, _ = input_var.shape
    # net = ReshapeLayer(net, (-1, feature_size))
    net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_size))
    for i in range(n_preproc_layers):
        net = Conv2DLayer(net, num_filters=n_preproc_units,
                          filter_size=5, pad='same', nonlinearity=elu)
        # net = DenseLayer(net, num_units=n_preproc_units, nonlinearity=elu)
        # if preproc_dropout > 0.:
        #     net = DropoutLayer(net, p=preproc_dropout)

    net = ReshapeLayer(net, (n_batch * n_time_steps, n_preproc_units * feature_size), name='bar')
    net = DenseLayer(net, num_units=n_combiner_units, nonlinearity=elu)
    net = ReshapeLayer(net, (n_batch, n_time_steps, n_combiner_units), name='foo')
    net = AverageLayer(net, axis=1, mask_input=mask)
    net = DenseLayer(net, num_units=24, nonlinearity=softmax)

    return net, input_var, mask_var


def build_rnn_model(feature_size,
                    n_preproc_layers,
                    n_preproc_units,
                    preproc_dropout,
                    n_combiner_units=12):

    from lasagne.layers import (DenseLayer, InputLayer, DropoutLayer,
                                ReshapeLayer, RecurrentLayer, ConcatLayer)
    from lasagne.nonlinearities import softmax, elu

    net = InputLayer((None, None, feature_size))
    mask = InputLayer((None, None))
    input_var = net.input_var
    mask_var = mask.input_var
    n_batch, n_time_steps, _ = input_var.shape

    if n_preproc_layers > 0:
        net = ReshapeLayer(net, (-1, feature_size))
        for i in range(n_preproc_layers):
            net = DenseLayer(net, num_units=n_preproc_units, nonlinearity=elu)
            if preproc_dropout > 0.:
                net = DropoutLayer(net, p=preproc_dropout)
        net = ReshapeLayer(net, (n_batch, n_time_steps, n_preproc_units))

    fwd = RecurrentLayer(
        incoming=net,
        mask_input=mask,
        num_units=n_combiner_units,
        W_in_to_hid=lnn.init.HeNormal(gain=0.9),
        W_hid_to_hid=np.identity(n_combiner_units, dtype=np.float32) * 0.3,
        learn_init=True,
        nonlinearity=elu,
        name='Recurrent Fwd',
        only_return_final=True,
        grad_clipping=1.
    )

    bck = RecurrentLayer(
        incoming=net,
        mask_input=mask,
        num_units=n_combiner_units,
        W_in_to_hid=lnn.init.HeNormal(gain=0.5),
        W_hid_to_hid=np.identity(n_combiner_units, dtype=np.float32) * 0.3,
        learn_init=True,
        nonlinearity=elu,
        name='Recurrent Bck',
        only_return_final=True,
        backwards=True,
        grad_clipping=1.
    )

    net = ConcatLayer([fwd, bck])
    net = DenseLayer(net, num_units=24, nonlinearity=softmax)

    return net, input_var, mask_var


def main():
    args = docopt(USAGE)
    n_preproc_layers = int(args['--n_preproc_layers'])
    n_preproc_units = int(args['--n_preproc_units'])
    preproc_dropout = float(args['--preproc_dropout'])
    combiner_type = args['--combiner_type']
    n_combiner_units = int(args['--n_combiner_units'])
    batch_size = int(args['--batch_size'])
    no_dist_sampling = args['--no_dist_sampling']
    n_epochs = int(args['--n_epochs'])
    exp_id = args['--exp_id']
    feature = args['--feature']
    data_type = args['--data']
    init_patience = int(args['--patience'])
    init_learn_rate = float(args['--init_lr'])

    print(args)

    exp_dir = join('results', exp_id)
    if not exists(exp_dir):
        os.makedirs(exp_dir)
    yaml.dump(args, open(join(exp_dir, 'config'), 'w'))

    if data_type == 'giantsteps':
        training_set, val_set, test_set, targ_dist = data.load_giantsteps(
            'data', 'feature_cache', feature, not no_dist_sampling
        )
    elif data_type == 'billboard':
        training_set, val_set, test_set, targ_dist = data.load_billboard(
            'data', 'feature_cache', feature, not no_dist_sampling
        )
    elif data_type == 'all':
        tr_gs, vl_gs, te_gs, targ_dist = data.load_giantsteps(
            'data', 'feature_cache', feature, not no_dist_sampling
        )
        tr_bb, vl_bb, te_bb, targ_dist = data.load_billboard(
            'data', 'feature_cache', feature, not no_dist_sampling
        )
        training_set = tr_gs + tr_bb
        val_set = vl_gs + vl_bb
        test_set = te_gs + te_bb
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    print('#Train: {}\n#Val: {}\n#Test: {}'.format(
        len(training_set), len(val_set), len(test_set))
    )

    if combiner_type == 'avg':
        build_model = build_avg_model
        learning_rate_schedule = {
            0: init_learn_rate,
            # 100: 0.0001,
            # 150: 0.00001,
            # 200: 0.000001
        }
    elif combiner_type == 'rnn':
        build_model = build_rnn_model
        learning_rate_schedule = {
            0:   0.0001,
            30:  0.00001,
            100: 0.000001,
            150: 0.0000001
        }
    else:
        raise ValueError('Unknown combiner type: {}'.format(combiner_type))

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

    loss += 1e-4 * lnn.regularization.regularize_network_params(
        model, lnn.regularization.l2)

    acc = tt.mean(lnn.objectives.categorical_accuracy(y_hat, y),
                  dtype='floatX')
    params = lnn.layers.get_all_params(model)
    learning_rate = theano.shared(np.array(learning_rate_schedule[0],
                                           dtype=theano.config.floatX))
    # updates = lnn.updates.rmsprop(loss, params, learning_rate=learning_rate)
    updates = lnn.updates.momentum(loss, params, learning_rate=learning_rate)
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
            batches.set_description('Loss: {:g}'.format(loss / n_it))
        return loss / n_it, acc / n_it

    print('{:>5s}{:>20s}{:>15s}{:>15s}{:>25s}{:>25s}'.format(
        'epoch', 'learning rate', 'loss', 'accuracy', 'validation loss',
        'validation accuracy')
    )

    best_val_acc = -np.inf
    train_log = []
    patience = init_patience
    best_params = None
    for epoch in tqdm.tqdm(range(n_epochs)):
        if epoch in learning_rate_schedule:
            learning_rate.set_value(learning_rate_schedule[epoch])

        train_loss, train_acc = iterate(train_it, train)
        val_loss, val_acc = iterate(val_it, test)

        tqdm.tqdm.write(
            '{:>5d}{:>20.12f}{:>15.6f}{:>15.6f}{:>25.6f}{:>25.6f}'.format(
                epoch, float(learning_rate.get_value()), float(train_loss),
                float(train_acc), float(val_loss), float(val_acc))
        )

        if val_acc > best_val_acc:
            patience = init_patience
            best_val_acc = val_acc
            best_params = lnn.layers.get_all_param_values(model)
            pickle.dump(
                best_params,
                open(join(exp_dir, 'best_model'), 'wb')
            )
        else:
            patience -= 1

        if patience == 0:
            tqdm.tqdm.write('Restarting with best...')
            lnn.layers.set_all_param_values(model, best_params)
            patience = init_patience
            lr = learning_rate.get_value()
            learning_rate.set_value(lr / 2.)

        train_log.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_accuracy': float(train_acc),
            'validation_loss': float(val_loss),
            'validation_accuracy': float(val_acc)
        })
        yaml.dump(train_log, open(join(exp_dir, 'log'), 'w'))

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

    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

if __name__ == '__main__':
    main()

