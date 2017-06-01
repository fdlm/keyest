import vamp
import numpy as np
from glob import glob
from multiprocessing import Pool
from os.path import basename, splitext, join
import madmom as mm
from itertools import imap, repeat
from docopt import docopt
from tqdm import tqdm

USAGE = """
Usage:
  test_qm.py [options] <key_dir> <dst_dir> <audio_files>...

Options:
  -f <fold_file>  test fold file
  -w <n_workers>  number of workers [default: 1]

"""

QM_KEY = 'qm-vamp-plugins:qm-keydetector'

KEYS = ['C major',
        'Db major',
        'D major',
        'Eb major',
        'E major',
        'F major',
        'F# major',
        'G major',
        'Ab major',
        'A major',
        'Bb major',
        'B major',
        'C minor',
        'C# minor',
        'D minor',
        'D# minor',
        'E minor',
        'F minor',
        'F# minor',
        'G minor',
        'G# minor',
        'A minor',
        'Bb minor',
        'B minor']


def global_key(vamp_result, signal_length):
    """
    return key with longest duration from result
    """
    starts = np.array([float(rd['timestamp'])
                       for rd in vamp_result['list']])
    ends = np.hstack([starts[1:], [signal_length]])
    dur = ends - starts
    keys = np.array([rd['values'][0]
                     for rd in vamp_result['list']], dtype=int) - 1
    ok = keys < 24
    return np.bincount(keys[ok], weights=dur[ok], minlength=24).argmax()


def compute_key(args):
    af, dst_dir = args
    sig = mm.audio.Signal(af, num_channels=1, dtype=float)
    key_class = global_key(
        vamp.collect(np.asarray(sig), sig.sample_rate, QM_KEY, output='key'),
        sig.length
    )
    key = KEYS[key_class]
    pred_file = join(dst_dir, splitext(basename(af))[0] + '.key.txt')
    with open(pred_file, 'w') as f:
        f.write(key)


def main():
    args = docopt(USAGE)
    dst_dir = args['<dst_dir>']
    n_workers = int(args['-w'])
    audio_files = args['<audio_files>']
    key_files = glob(args['<key_dir>'] + '/*.key')
    split_file = args['-f']

    key_songs = [splitext(basename(kf))[0] for kf in key_files]

    # use only audio files for which we have a key
    audio_files = [af for af in audio_files
                   if splitext(basename(af))[0] in key_songs]
    # use only audio files defined in the split file
    if split_file:
        splits = [l.strip() for l in open(split_file)]
        audio_files = [af for af in audio_files
                       if splitext(basename(af))[0] in splits]

    if n_workers > 1:
        p = Pool(n_workers)
        imap_fun = p.imap
    else:
        imap_fun = imap

    with tqdm(total=len(audio_files)) as progress_bar:
        process_data = zip(audio_files, repeat(dst_dir))
        for _ in imap_fun(compute_key, process_data):
            progress_bar.update()

if __name__ == "__main__":
    main()


