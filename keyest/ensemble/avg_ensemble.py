import numpy as np
import os
import shutil
import yaml
from os.path import splitext, basename, join
from glob import glob
from docopt import docopt
from tqdm import tqdm

from keyest.config import EXPERIMENT_ROOT


USAGE = """
Usage:
  avg_ensemble.py [--force] [--save_preds] [--temperature=<f>] [--with_train]
                  <ensemble_exp_id> <exp_ids>...
"""


KEYS = ['A major', 'Bb major', 'B major', 'C major', 'Db major', 'D major',
        'Eb major', 'E major', 'F major', 'F# major', 'G major', 'Ab major',
        'A minor', 'Bb minor', 'B minor', 'C minor', 'C# minor', 'D minor',
        'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor']


class TemperatureSoftmax(object):

    def __init__(self, temperature):
        self.temperature = temperature

    def __call__(self, x):
        x = x / self.temperature
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)


def main():
    args = docopt(USAGE)

    ensemble_dir = join(EXPERIMENT_ROOT, args['<ensemble_exp_id>'])
    if os.path.exists(ensemble_dir):
        if not args['--force']:
            print('ERROR: experiment directory already exists!')
            return 1
        else:
            shutil.rmtree(ensemble_dir)
    os.makedirs(ensemble_dir)
    temperature = float(args['--temperature']) if args['--temperature'] else 1.0
    config = dict(
        model='avg_ensemble',
        temperature=temperature,
        source_experiments=args['<exp_ids>']
    )
    yaml.dump(config, open(join(ensemble_dir, 'config.yaml'), 'w'))

    softmax = TemperatureSoftmax(temperature)
    first_exp_dir = join(EXPERIMENT_ROOT, args['<exp_ids>'][0])
    setups = ['val', 'test']
    if args['--with_train']:
        setups.insert(0, 'train')
    for setup in setups:
        os.makedirs(join(ensemble_dir, setup))
        for f in tqdm(glob(join(first_exp_dir, setup, '*.npy')), desc=setup):
            preds = [softmax(np.load(f))]
            fname = splitext(basename(f))[0]
            for exp_id in args['<exp_ids>'][1:]:
                preds.append(np.load(
                    join(EXPERIMENT_ROOT, exp_id, setup, fname + '.npy')))
                pred = np.vstack(preds).mean(axis=0)
                if args['--save_preds']:
                    np.save(join(ensemble_dir, setup, fname), pred)
                with open(join(ensemble_dir, setup, fname + '.key.txt'), 'w') as f:
                    f.write(KEYS[pred.argmax()])


if __name__ == "__main__":
    main()
