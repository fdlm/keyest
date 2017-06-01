import sys
from collections import OrderedDict
from glob import glob
import madmom as mm

from os.path import splitext

USAGE = """
Usage:
  eval.py <files>...
"""


KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'cb': 11}


def load_key(key_file):
    tonic, mode = open(key_file).read().strip().split()
    key_class = KEY_TO_SEMITONE[tonic.lower()]
    if mode == 'minor':
        key_class += 12
    elif mode == 'major':
        key_class += 0
    else:
        raise ValueError(mode)

    return key_class


def error_type(ref_key_class, pred_key_class):
    ref_root = ref_key_class % 12
    ref_mode = ref_key_class // 12
    pred_root = pred_key_class % 12
    pred_mode = pred_key_class // 12
    major = 0
    minor = 1

    if pred_root == ref_root and pred_mode == ref_mode:
        return 'correct', 1.0
    # correct, but pessimistic
    # if pred_mode == ref_mode and ((pred_root - ref_root) % 12 == 7):
    #     return 'fifth', 0.5
    # if pred_mode == ref_mode and ((pred_root - ref_root) % 12 == 5):
    #     return 'fifth', 0.5

    # wrong, but for the sake of comparison
    if (pred_root - ref_root) % 12 == 7:
        return 'fifth', 0.5
    if (pred_root - ref_root) % 12 == 5:
        return 'fifth', 0.5
    if (ref_mode == major and pred_mode != ref_mode and (
                (pred_root - ref_root) % 12 == 9)):
        return 'relative', 0.3
    if (ref_mode == minor and pred_mode != ref_mode and (
                (pred_root - ref_root) % 12 == 3)):
        return 'relative', 0.3
    if pred_mode != ref_mode and pred_root == ref_root:
        return 'parallel', 0.2
    else:
        return 'error', 0.0


def main():
    if len(sys.argv) < 2:
        print USAGE
        return 1

    files = sys.argv[1:]
    det_files = [f for f in files if splitext(f)[1] == '.txt']
    ann_files = [mm.utils.match_file(f, files, '.txt', '.key')[0]
                 for f in det_files]
    assert len(ann_files) == len(det_files)

    results = OrderedDict([
        ('correct', 0.0),
        ('fifth', 0.0),
        ('relative', 0.0),
        ('parallel', 0.0),
        ('error', 0.0),
        ('weighted', 0.0)
    ])

    for ann_f, det_f in zip(ann_files, det_files):
        ann_key = load_key(ann_f)
        det_key = load_key(det_f)
        error, weight = error_type(ann_key, det_key)
        results[error] += 1.
        results['weighted'] += weight

    print('Evaluated {} files.'.format(len(det_files)))
    for k, v in results.items():
        print('{:>15s} = {:5.2f}'.format(k, 100. * v / len(ann_files)))

if __name__ == '__main__':
    main()
