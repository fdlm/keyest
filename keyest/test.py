from __future__ import print_function
import yaml
import numpy as np
from os.path import join
from docopt import docopt
from termcolor import colored
from tqdm import tqdm

import data
import models


USAGE = """
Usage:
    test.py --exp_dir=<S> --data=<S>
"""


KEYS = ['A major', 'Bb major', 'B major', 'C major', 'Db major', 'D major',
        'Eb major', 'E major', 'F major', 'F# major', 'G major', 'Ab major',
        'A minor', 'Bb minor', 'B minor', 'C minor', 'C# minor', 'D minor',
        'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor']


def test(model, data_set, dst_dir):
    for piece in tqdm(data_set.datasources, desc='Predicting'):
        piece_data = piece[:][0]
        mask = np.ones((1, piece_data.shape[1]), dtype=np.float32)
        predictions = model.process(piece_data, mask)
        pred_file = join(dst_dir, piece.name)
        np.save(pred_file, predictions)
        with open(join(dst_dir, piece.name + '.key.txt'), 'w') as f:
            f.write(KEYS[predictions.argmax()])


def main():
    args = docopt(USAGE)
    config = yaml.load(open(join(args['--exp_dir'], 'config.yaml')))

    # -------------------------
    # Load data and build model
    # -------------------------

    if args['--data'] == 'all':
        ds = 'giantsteps,billboard,musicnet,cmdb'
    else:
        ds = args['--data']

    _, _, test_set = data.load(
        datasets=ds,
        **config['data_params']
    )

    print(colored('\nData:\n', color='blue'))
    print('Test Set: ', len(test_set))

    model = models.build_model(config['model'],
                               feature_shape=test_set.dshape,
                               **config['model_params'])
    model.load(join(args['--exp_dir'], 'best_model.pkl'))

    # -------------------
    # Compute predictions
    # -------------------
    print(colored('\nApplying on Test Set:\n', color='blue'))
    test(model, test_set, args['--exp_dir'])


if __name__ == "__main__":
    main()
