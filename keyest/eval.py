import sys
from glob import glob
from itertools import combinations
from scipy.stats import wilcoxon, binom_test

import madmom as mm

from docopt import docopt
from collections import OrderedDict, Counter
from os.path import splitext, basename, join

USAGE = """
Usage:
  eval.py single <files>...
  eval.py compare <pattern> <gt_dir> <directories>...
"""


KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd': 2, 'd#': 3, 'eb': 3, 'e': 4,
                   'f': 5, 'f#': 6, 'gb': 6, 'g': 7, 'g#': 8, 'ab': 8, 'a': 9,
                   'a#': 10, 'bb': 10, 'b': 11, 'cb': 11}


def load_key(key_file):
    tonic, mode = open(key_file).read().strip().split()
    key_class = KEY_TO_SEMITONE[tonic.lower()]
    if mode in ['minor', 'min']:
        key_class += 12
    elif mode in ['major', 'maj']:
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
    if pred_mode == ref_mode and ((pred_root - ref_root) % 12 == 7):
        return 'fifth', 0.5
    if pred_mode == ref_mode and ((pred_root - ref_root) % 12 == 5):
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


def collect_results(ann_files, det_files):
    results = dict(song=[], error=[], weight=[])
    for ann_f, det_f in zip(ann_files, det_files):
        ann_key = load_key(ann_f)
        det_key = load_key(det_f)
        error, weight = error_type(ann_key, det_key)
        results['song'].append(splitext(basename(ann_f))[0])
        results['error'].append(error)
        results['weight'].append(weight)
    return results


def average_results(results):
    n = len(results['song'])
    c = Counter(results['error'])

    return OrderedDict([
        ('correct', float(c['correct']) / n),
        ('fifth', float(c['fifth']) / n),
        ('relative', float(c['relative']) / n),
        ('parallel', float(c['parallel']) / n),
        ('error', float(c['error']) / n),
        ('weighted', sum(results['weight']) / n)
    ])


def to_ranks(weights):
    ranks = [1.0, 0.5, 0.3, 0.2, 0.0]
    return [ranks.index(w) + 1 for w in weights]


def main():
    args = docopt(USAGE)

    if args['single']:
        files = sys.argv[1:]
        det_files = mm.utils.filter_files(files, '.key.txt')
        ann_files = [mm.utils.match_file(f, files, '.key.txt', '.key')[0]
                     for f in det_files]
        assert len(ann_files) == len(det_files)

        results = average_results(collect_results(ann_files, det_files))

        print('Evaluated {} files.'.format(len(det_files)))
        for k, v in results.items():
            print('{:>15s} = {:5.2f}'.format(k, 100. * v))
    else:
        det_files = {}
        for det_dir in args['<directories>']:
            name = det_dir.split('/')[-1]
            if name == 'predictions':
                name = det_dir.split('/')[-2]
            df = glob(join(det_dir, args['<pattern>'] + '.key.txt'))
            df.sort()
            det_files[name] = df

        ann_files = mm.utils.search_files(args['<gt_dir>'], '.key')
        ann_files = [mm.utils.match_file(f, ann_files, '.key.txt', '.key')[0]
                     for f in df]

        for name in det_files:
            assert len(ann_files) == len(det_files[name])

        results = {name: collect_results(ann_files, det_files[name])
                   for name in det_files}
        avg_res = {name: average_results(results[name])
                   for name in results}

        for n1, n2 in combinations(results.keys(), 2):
            r1 = to_ranks(results[n1]['weight'])
            r2 = to_ranks(results[n2]['weight'])
            print '{} vs. {}: {:5.2f} / {:5.2f}, p = {:.3f}'.format(
                n1, n2,
                avg_res[n1]['weighted'] * 100, avg_res[n2]['weighted'] * 100,
                wilcoxon(r1, r2).pvalue
            )


if __name__ == '__main__':
    main()
