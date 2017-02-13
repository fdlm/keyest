import madmom as mm
import numpy as np
import dmgr
import string
from os.path import join


class SingleKeyMajMinTarget(object):

    def __init__(self):
        natural = zip(string.uppercase[:7], [0, 2, 3, 5, 7, 8, 10])
        sharp = map(lambda v: (v[0] + '#', (v[1] + 1) % 12), natural)
        flat = map(lambda v: (v[0] + 'b', (v[1] - 1) % 12), natural)
        self.root_note_map = dict(natural + sharp + flat)

    @property
    def name(self):
        return 'single_key_majmin'

    def __call__(self, target_file, _):
        root, mode = open(target_file).read().split()
        target_class = self.root_note_map[root]
        if mode == 'minor':
            target_class += 12
        return np.int32(target_class)


class LogFiltSpec:

    def __init__(self, frame_sizes, num_bands, fmin, fmax, fps, unique_filters,
                 sample_rate=44100, fold=None):

        self.frame_sizes = frame_sizes
        self.num_bands = num_bands
        self.fmax = fmax
        self.fmin = fmin
        self.fps = fps
        self.unique_filters = unique_filters
        self.sample_rate = sample_rate

    @property
    def name(self):
        return 'lfs_fps={}_num-bands={}_fmin={}_fmax={}_frame_sizes=[{}]'.format(
                self.fps, self.num_bands, self.fmin, self.fmax,
                '-'.join(map(str, self.frame_sizes))
        ) + ('_uf' if self.unique_filters else '')

    def __call__(self, audio_file):
        # do not resample because ffmpeg/avconv creates terrible sampling
        # artifacts
        specs = [
            mm.audio.spectrogram.LogarithmicFilteredSpectrogram(
                audio_file, num_channels=1, sample_rate=self.sample_rate,
                fps=self.fps, frame_size=ffts,
                num_bands=self.num_bands, fmin=self.fmin, fmax=self.fmax,
                unique_filters=self.unique_filters)
            for ffts in self.frame_sizes
        ]

        return np.hstack(specs).astype(np.float32)


def load_giantsteps_key_dataset(data_dir, feature_cache_dir):

    compute_features = LogFiltSpec(
        frame_sizes=[8192],
        num_bands=24,
        fmin=65,
        fmax=2100,
        fps=5,
        unique_filters=True
    )

    compute_targets = SingleKeyMajMinTarget()

    return dmgr.Dataset(
        data_dir=data_dir,
        feature_cache_dir=feature_cache_dir,
        split_defs=[join(data_dir, 'splits', '8-fold_cv_key_{}.fold'.format(i))
                    for i in range(8)],
        source_ext='.mp3',
        gt_ext='.key',
        compute_features=compute_features,
        compute_targets=compute_targets
    )


def get_splits(dataset, val_fold, test_fold):
    train_split, val_split, test_split = dataset.fold_split(val_fold,
                                                            test_fold)
    train_set = load_data(train_split, use_augmented=True)
    val_set = load_data(val_split, use_augmented=False)
    test_set = load_data(test_split, use_augmented=False)

    l = [np.load(f) for f in train_split['targ'] if '.0.' in f]
    train_targ_dist = np.bincount(np.hstack(l), minlength=24).astype(np.float)
    train_targ_dist /= train_targ_dist.sum()

    return train_set, val_set, test_set, train_targ_dist


def load_data(files, use_augmented):
    return [(np.load(f[0]), np.load(f[1]))
            for f in zip(files['feat'], files['targ'])
            if use_augmented or '.0.' in f[0]]
