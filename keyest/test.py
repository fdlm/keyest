from __future__ import print_function
import yaml
import numpy as np
from os.path import join
from docopt import docopt
from termcolor import colored
from tqdm import tqdm
import os

import data
import models


USAGE = """
Usage:
    test.py --exp_dir=<S> --data=<S> [--save_pred] [--proc_aug]
"""


KEYS = ['A major', 'Bb major', 'B major', 'C major', 'Db major', 'D major',
        'Eb major', 'E major', 'F major', 'F# major', 'G major', 'Ab major',
        'A minor', 'Bb minor', 'B minor', 'C minor', 'C# minor', 'D minor',
        'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor']


def test(model, datasources, dst_dir, save_pred=True):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for piece in tqdm(datasources, desc='Predicting'):
        piece_data = piece[:][:model.n_features]
        if model.needs_mask:
            mask = np.ones((1, piece_data[0].shape[1]), dtype=np.float32)
            piece_data += (mask,)
        predictions = model.process(*piece_data)
        if save_pred:
            pred_file = join(dst_dir, piece.name)
            np.save(pred_file, predictions)
        with open(join(dst_dir, piece.name + '.key.txt'), 'w') as f:
            f.write(KEYS[predictions.argmax()])


def main():
    args = docopt(USAGE)
    save_pred = args['--save_pred']
    experiment_dir = args['--exp_dir']
    config = yaml.load(open(join(experiment_dir, 'config.yaml')))

    # -------------------------
    # Load data and build model
    # -------------------------

    if args['--data'] == 'all':
        ds = 'giantsteps,billboard,musicnet,cmdb'
    else:
        ds = args['--data']

    train_set, val_set, test_set = data.load(
        datasets=ds,
        **config['data_params']
    )

    Model = models.get_model(config['model'])
    train_src, val_src, test_src = data.create_datasources(
        datasets=[train_set, val_set, test_set],
        representations=(Model.source_representations() +
                         Model.target_representations())
    )

    print(colored('\nData:\n', color='blue'))
    print('Training Set: ', len(train_src))
    print('Validation Set: ', len(val_src))
    print('Test Set: ', len(test_src))

    model = Model(feature_shape=test_src.dshape, **config['model_params'])
    model.load(join(args['--exp_dir'], 'best_model.pkl'))

    # -------------------
    # Compute predictions
    # -------------------

    print(colored('\nApplying on Training Set:\n', color='blue'))
    if not args['--proc_aug']:
        train_ds = [t for t in train_src.datasources if '.0' in t.name]
    else:
        train_ds = train_src.datasources
    test(model, train_ds, join(experiment_dir, 'train'), save_pred)
    print(colored('\nApplying on Validation Set:\n', color='blue'))
    test(model, val_src.datasources,
         join(experiment_dir, 'val'), save_pred)
    print(colored('\nApplying on Test Set:\n', color='blue'))
    test(model, test_src.datasources,
         join(experiment_dir, 'test'), save_pred)


if __name__ == "__main__":
    main()
