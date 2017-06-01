from __future__ import print_function

import os.path
import pickle
from os.path import join, basename, splitext

import numpy as np
import theano
import yaml
from docopt import docopt
from tqdm import tqdm

import data
import lasagne as lnn

USAGE = """
Usage:
    test.py <exp_id> <data_set>
"""

KEYS = ['A major',
        'Bb major',
        'B major',
        'C major',
        'Db major',
        'D major',
        'Eb major',
        'E major',
        'F major',
        'F# major',
        'G major',
        'Ab major',
        'A minor',
        'Bb minor',
        'B minor',
        'C minor',
        'C# minor',
        'D minor',
        'D# minor',
        'E minor',
        'F minor',
        'F# minor',
        'G minor',
        'G# minor']


def create_key_estimator(exp_dir, config, feature_size):
    if config['--combiner_type'] == 'avg':
        from train_theano import build_avg_model as build_model
    elif config['--combiner_type'] == 'rnn':
        from train_theano import build_rnn_model as build_model
    else:
        raise ValueError('Unknown combiner model: {}'.format(
            config['--combiner_type']))

    model, X, m = build_model(
        feature_size=feature_size,
        n_preproc_layers=int(config['--n_preproc_layers']),
        n_preproc_units=int(config['--n_preproc_units']),
        preproc_dropout=float(config['--preproc_dropout']),
        n_combiner_units=int(config['--n_combiner_units'])
    )
    lnn.layers.set_all_param_values(
        model,
        pickle.load(open(join(exp_dir, 'best_model'), 'rb'))
    )
    y = lnn.layers.get_output(model, deterministic=True)
    process = theano.function(
        inputs=[X, m],
        outputs=y.argmax(),
    )

    def proc(data):
        return process(
            data[None, ...], np.ones((1, len(data)), dtype=np.float32)
        )

    return proc


def main():
    args = docopt(USAGE)
    exp_dir = join('results', args['<exp_id>'])
    config = yaml.load(open(join(exp_dir, 'config')))

    if args['<data_set>'] == 'giantsteps':
        test_dataset = data.load_giantsteps_key_dataset(
            data_dir=join('data', 'giantsteps-key-dataset'),
            feature_cache_dir='feature_cache',
            feature=config['--feature']
        )

        test_files = test_dataset.all_files()
        # no augmented files in giantsteps dataset anyways, keep flag true
        test_set = data.load_data(test_files, use_augmented=True)
        songs = ['.'.join(basename(tf).split('.')[:2])
                 for tf in test_files['feat']]
    elif args['<data_set>'] == 'billboard':
        test_dataset = data.load_billboard_key_dataset(
            data_dir=join('data', 'mcgill-billboard-augmented'),
            feature_cache_dir='feature_cache',
            feature=config['--feature']
        )
        _, _, test_files = test_dataset.fold_split(0, 1)
        test_set = data.load_data(test_files, use_augmented=False)
        songs = [basename(tf).split('.')[0]
                 for tf in test_files['feat']
                 if '.0.' in tf]
    else:
        raise ValueError('Unknown data set: {}'.format(args['<data_type>']))

    compute_key = create_key_estimator(
        exp_dir, config, test_set[0][0].shape[-1])

    test_result_dir = join(exp_dir, 'predictions')
    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    for song, (test_data, _) in tqdm(zip(songs, test_set)):
        pred = compute_key(test_data)
        key = KEYS[pred]
        pred_file = join(test_result_dir, song + '.key.txt')
        with open(pred_file, 'w') as f:
            f.write(key)


if __name__ == "__main__":
    main()
