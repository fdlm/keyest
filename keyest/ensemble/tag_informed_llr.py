from __future__ import print_function
"""
TODO:
 - collect prediction files of N classifiers
 - load tags for each of the files
 - split these prediction files into train and test sets
 - formulate LLR for score fusion (1)
 - formulate tag-informed LLR for score fusion (2)
 - formulate simple averaging (this is a specific parameter setting for (1)) (3)
"""
import os
import shutil
import numpy as np
import theano
import theano.tensor as tt
import lasagne as lnn
import pickle
from os.path import join
from termcolor import colored
from docopt import docopt
from tqdm import tqdm
from keyest.models import add_dimension

import auds
from auds.representations import SingleKeyMajMin, Precomputed

import keyest.data
from keyest.config import EXPERIMENT_ROOT, CACHE_DIR
from keyest.test import KEYS


USAGE = """
Usage:
    tag_informed_llr.py [--simple] [--force] [--n_epochs=<I>] [--patience=<I>] --data=<S> --out=<S> <prediction_dirs>...
    
Arguments:
    <prediction_dirs>  Directories with key predictions

Options:
    --simple  Use simple LLR fusion, not tag-informed
    --data=<S>  Dataset(s) to load (comma separated if multiple)
    --out=<S>  Experiment output directory
    --force  Overwrite experiment output directory if it already exists.
    --n_epochs=<I>  Max number of epochs to train [default: 100000]
    --patience=<I>  Early stopping patience [default: 1000]
"""


class Args(object):
    def __init__(self):
        args = docopt(USAGE)
        self.data = args['--data']
        self.out = args['--out']
        self.prediction_dirs = args['<prediction_dirs>']
        self.force = args['--force']
        self.n_epochs = int(args['--n_epochs'])
        self.patience = int(args['--patience'])
        self.simple = args['--simple']


def test(process, data_src, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for piece in tqdm(data_src.datasources, desc='Predicting'):
        piece_data = piece[:][:-1]  # remove target

        # last index are tags
        if piece_data[-1].shape[1] != 24:
            process_data = [np.array(piece_data[:-1]).transpose(1, 0, 2),
                            np.array(piece_data[-1])]
        else:
            process_data = [np.array(piece_data).transpose(1, 0, 2)]

        predictions = process(*process_data)[0]
        pred_file = join(dst_dir, piece.name)
        np.save(pred_file, predictions)
        with open(join(dst_dir, piece.name + '.key.txt'), 'w') as f:
            f.write(KEYS[predictions.argmax()])


def main():
    args = Args()

    experiment_dir = join(EXPERIMENT_ROOT, args.out)
    if os.path.exists(experiment_dir):
        if not args.force:
            print('ERROR: experiment directory already exists!')
            return 1
        else:
            shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir)

    if args.data == 'all':
        ds = 'giantsteps,billboard,musicnet,cmdb'
    else:
        ds = args.data

    train_set, val_set, test_set = keyest.data.load(datasets=ds)
    target_representation = auds.representations.make_cached(SingleKeyMajMin(),
                                                             CACHE_DIR)

    train_set = val_set
    source_representations = [
        Precomputed(
            [join(src_dir, setup) for setup in ['train', 'val', 'test']],
            view='audio', name='key_classification'
        )
        for src_dir in args.prediction_dirs
    ]

    if not args.simple:
        source_representations.append(add_dimension(
            Precomputed('/home/filip/.tmp/jamendo_tags', 'audio',
                        'jamendo_tags')
        ))

    train_src, val_src, test_src = keyest.data.create_datasources(
        datasets=[train_set, val_set, test_set],
        representations=source_representations + [target_representation]
    )

    print(colored('\nData:\n', color='blue'))
    print('Train Set:       ', len(train_src))
    print('Validation Set:  ', len(val_src))
    print('Test Set:        ', len(test_src))

    n_models = len(args.prediction_dirs)
    x = tt.ftensor3('x')
    y = tt.fmatrix('y')

    if args.simple:
        W = theano.shared(np.ones((1, n_models)).astype(np.float32), name='W')
        b = theano.shared(np.zeros(24).astype(np.float32), name='b')
        y_hat = tt.nnet.softmax(W.dot(x)[0] + b)
        params = [W, b]
        loss = tt.nnet.categorical_crossentropy(y_hat, y).mean()
        updates = lnn.updates.sgd(loss, params, learning_rate=1.)
        train = theano.function(inputs=[x, y], outputs=loss, updates=updates)
        evaluate = theano.function(inputs=[x, y], outputs=loss)
        process = theano.function(inputs=[x], outputs=y_hat)

        # train_data = val_src[:]
        train_data = train_src[:]
        train_preds = np.array(train_data[:-1]).transpose(1, 0, 2)
        train_y = np.array(train_data[-1])
        train_batch = [train_preds, train_y]

        val_data = val_src[:]
        val_preds = np.array(val_data[:-1]).transpose(1, 0, 2)
        val_y = np.array(val_data[-1])
        val_batch = [val_preds, val_y]

    else:
        n_tags = train_src[0][-2].shape[0]
        t = tt.fmatrix('t')
        W_c = theano.shared(
            np.random.normal(scale=0.1, size=(n_models, n_tags)).astype(np.float32), name='W_t')
        b_c = theano.shared(np.zeros(n_models, dtype=np.float32), name='b_t')
        W_k = theano.shared(
            np.random.normal(scale=0.1, size=(24, n_tags)).astype(np.float32), name='W_b')
        b_k = theano.shared(np.zeros(24, dtype=np.float32), name='b_b')

        W_t = W_c.dot(t.T).T + b_c
        b_t = W_k.dot(t.T).T + b_k

        # There has to be a better way!
        y_hat = tt.nnet.softmax(
            (W_t.dimshuffle(0, 1, 'x') * x).sum(1) + b_t
        )
        params = [W_c, b_c, W_k, b_k]

        loss = tt.nnet.categorical_crossentropy(y_hat, y).mean()
        updates = lnn.updates.sgd(loss, params, learning_rate=0.1)
        train = theano.function(inputs=[x, t, y], outputs=loss,
                                updates=updates)
        evaluate = theano.function(inputs=[x, t, y], outputs=loss)
        process = theano.function(inputs=[x, t], outputs=y_hat)

        train_data = train_src[:]
        train_preds = np.array(train_data[:-2]).transpose(1, 0, 2)
        train_tags = np.array(train_data[-2])
        train_y = np.array(train_data[-1])
        train_batch = [train_preds, train_tags, train_y]

        val_data = val_src[:]
        val_preds = np.array(val_data[:-2]).transpose(1, 0, 2)
        val_tags = np.array(val_data[-2])
        val_y = np.array(val_data[-1])
        val_batch = [val_preds, val_tags, val_y]

    l_prev = np.inf
    wait = 0

    epochs = tqdm(range(args.n_epochs))
    for _ in epochs:
        train_loss = train(*train_batch)
        val_loss = evaluate(*val_batch)

        epochs.set_description('Loss: {:.5f}    Val Loss: {:.5f}'.format(
            float(train_loss), float(val_loss)))
        if val_loss < l_prev:
            l_prev = val_loss
            wait = 0
        else:
            wait += 1
            if wait > args.patience:
                break

    pickle.dump([p.get_value() for p in params],
                open(join(experiment_dir, 'llr_params.pkl'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    test(process, train_src, join(experiment_dir, 'train'))
    test(process, val_src, join(experiment_dir, 'val'))
    test(process, test_src, join(experiment_dir, 'test'))


if __name__ == "__main__":
    main()
