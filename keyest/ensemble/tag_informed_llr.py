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

from auds.representations import SingleKeyMajMin, Precomputed, make_cached

import keyest.data
from keyest.config import EXPERIMENT_ROOT, CACHE_DIR


USAGE = """
Usage:
    tag_informed_llr.py [--simple] [--force] [--n_epochs=<I>] [--patience=<I>] 
                        [--train_on_val] [--n_tags=<I>] --data=<S> --out=<S> 
                        <exp_ids>...
    
Arguments:
    <prediction_dirs>  Directories with key predictions

Options:
    --simple  Use simple LLR fusion, not tag-informed
    --data=<S>  Dataset(s) to load (comma separated if multiple)
    --out=<S>  Experiment output directory
    --force  Overwrite experiment output directory if it already exists.
    --n_epochs=<I>  Max number of epochs to train [default: 100000]
    --patience=<I>  Early stopping patience [default: 1000]
    --train_on_val  Use validation set for training
    --n_tags=<I>  Number of tags to use (PCA projected) [default: 65]
"""


class Args(object):
    def __init__(self):
        args = docopt(USAGE)
        self.data = args['--data']
        self.out = args['--out']
        self.experiment_ids = args['<exp_ids>']
        self.force = args['--force']
        self.n_epochs = int(args['--n_epochs'])
        self.patience = int(args['--patience'])
        self.simple = args['--simple']
        self.train_on_val = args['--train_on_val']
        self.n_tags = int(args['--n_tags'])


def test(process, data_src, dst_dir, pca=None):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for piece in tqdm(data_src.datasources, desc='Predicting'):
        piece_data = piece[:][:-1]  # remove target

        # last index are tags
        if piece_data[-1].shape[1] != 24:
            process_data = [np.array(piece_data[:-1]).transpose(1, 0, 2),
                            np.array(piece_data[-1])]
            if pca:
                process_data[-1] = pca.transform(process_data[-1]).astype(np.float32)
        else:
            process_data = [np.array(piece_data).transpose(1, 0, 2)]

        predictions = process(*process_data)[0]
        pred_file = join(dst_dir, piece.name)
        np.save(pred_file, predictions)
        with open(join(dst_dir, piece.name + '.key.txt'), 'w') as f:
            SingleKeyMajMin().map_back(predictions)


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
    target_representation = make_cached(SingleKeyMajMin(), CACHE_DIR)
    if args.train_on_val:
        train_set = val_set

    source_representations = [
        Precomputed(
            [join(EXPERIMENT_ROOT, exp_id, setup) for setup in ['train', 'val', 'test']],
            view='audio', name='key_classification'
        )
        for exp_id in args.experiment_ids
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

    n_models = len(args.experiment_ids)
    x = tt.ftensor3('x')
    y = tt.fmatrix('y')

    if args.simple:
        W = theano.shared(np.ones((1, n_models), dtype=np.float32), name='W')
        b = theano.shared(np.zeros(24, dtype=np.float32), name='b')
        temp = theano.shared(np.ones((1, n_models, 1), dtype=np.float32),
                             broadcastable=(True, False, True), name='T')
        y_hat = tt.nnet.softmax(W.dot(x / temp)[0] + b)
        params = [W, b, temp]
        loss = tt.nnet.categorical_crossentropy(y_hat, y).mean()
        updates = lnn.updates.adam(loss, params, learning_rate=0.1)
        train = theano.function(inputs=[x, y], outputs=loss, updates=updates)
        evaluate = theano.function(inputs=[x, y], outputs=loss)
        process = theano.function(inputs=[x], outputs=y_hat)

        train_data = train_src[:]
        train_preds = np.array(train_data[:-1]).transpose(1, 0, 2)
        train_y = np.array(train_data[-1])
        train_batch = [train_preds, train_y]

        val_data = val_src[:]
        val_preds = np.array(val_data[:-1]).transpose(1, 0, 2)
        val_y = np.array(val_data[-1])
        val_batch = [val_preds, val_y]
        pca = None

    else:
        n_tags = args.n_tags
        t = tt.fmatrix('t')
        W_c = theano.shared(
            np.random.normal(scale=0.1, size=(n_models, n_tags)).astype(np.float32), name='W_t')
        b_c = theano.shared(np.zeros(n_models, dtype=np.float32), name='b_t')
        W_k = theano.shared(
            np.random.normal(scale=0.1, size=(24, n_tags)).astype(np.float32), name='W_b')
        b_k = theano.shared(np.zeros(24, dtype=np.float32), name='b_b')
        temp = theano.shared(np.ones((1, n_models, 1), dtype=np.float32),
                             broadcastable=(True, False, True), name='T')
        W_t = W_c.dot(t.T).T + b_c
        b_t = W_k.dot(t.T).T + b_k
        # There has to be a better way!
        y_hat = tt.nnet.softmax(
            (W_t.dimshuffle(0, 1, 'x') * x / temp).sum(1) + b_t
        )
        params = [W_c, b_c, W_k, b_k, temp]

        loss = tt.nnet.categorical_crossentropy(y_hat, y).mean()
        updates = lnn.updates.adam(loss, params, learning_rate=0.01)
        train = theano.function(inputs=[x, t, y], outputs=loss,
                                updates=updates)
        evaluate = theano.function(inputs=[x, t, y], outputs=loss)
        process = theano.function(inputs=[x, t], outputs=y_hat)

        train_data = train_src[:]
        train_preds = np.array(train_data[:-2]).transpose(1, 0, 2)
        train_tags = np.array(train_data[-2])
        from sklearn.decomposition import PCA
        pca = PCA(n_tags).fit(train_tags)
        train_tags = pca.transform(train_tags).astype(np.float32)
        train_y = np.array(train_data[-1])
        train_batch = [train_preds, train_tags, train_y]

        val_data = val_src[:]
        val_preds = np.array(val_data[:-2]).transpose(1, 0, 2)
        val_tags = pca.transform(np.array(val_data[-2])).astype(np.float32)
        val_y = np.array(val_data[-1])
        val_batch = [val_preds, val_tags, val_y]

    l_prev = np.inf
    wait = 0

    epochs = tqdm(range(args.n_epochs))
    try:
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
    except KeyboardInterrupt:
        pass

    pickle.dump([p.get_value() for p in params],
                open(join(experiment_dir, 'llr_params.pkl'), 'wb'),
                protocol=pickle.HIGHEST_PROTOCOL)

    test(process, train_src, join(experiment_dir, 'train'), pca)
    test(process, val_src, join(experiment_dir, 'val'), pca)
    test(process, test_src, join(experiment_dir, 'test'), pca)


if __name__ == "__main__":
    main()
