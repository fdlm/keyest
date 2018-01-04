import numpy as np
from os.path import splitext, basename, join
from glob import glob
from docopt import docopt
from tqdm import tqdm

USAGE = """
Usage:
  avg_ensemble.py <dst_dir> <exp_dirs>...
"""


KEYS = ['A major', 'Bb major', 'B major', 'C major', 'Db major', 'D major',
        'Eb major', 'E major', 'F major', 'F# major', 'G major', 'Ab major',
        'A minor', 'Bb minor', 'B minor', 'C minor', 'C# minor', 'D minor',
        'D# minor', 'E minor', 'F minor', 'F# minor', 'G minor', 'G# minor']


def main():
    args = docopt(USAGE)
    for f in tqdm(glob(join(args['<exp_dirs>'][0], '*.npy'))):
        preds = [np.load(f)]
        fname = splitext(basename(f))[0]
        for ed in args['<exp_dirs>'][1:]:
            preds.append(np.load(join(ed, fname + '.npy')))
        pred = np.vstack(preds).mean(axis=0)
        np.save(join(args['<dst_dir>'], fname), pred)
        with open(join(args['<dst_dir>'], fname + '.key.txt'), 'w') as f:
            f.write(KEYS[pred.argmax()])


if __name__ == "__main__":
    main()
