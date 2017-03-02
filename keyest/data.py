from __future__ import print_function
import madmom as mm
import numpy as np
import dmgr
import string
from os.path import join


class SingleKeyMajMinTarget(object):

    def __init__(self):
        natural = zip(string.uppercase[:7], [0, 2, 3, 5, 7, 8, 10])
        sharp = map(lambda v: (v[0] + '#', (v[1] + 1) % 12), natural)
        flat = map(lambda v: (v[0] + 'B', (v[1] - 1) % 12), natural)
        self.root_note_map = dict(natural + sharp + flat)

    @property
    def name(self):
        return 'single_key_majmin'

    def __call__(self, target_file, _):
        root, mode = open(target_file).read().split()
        target_class = self.root_note_map[root.upper()]
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


class DeepChroma:

    def __init__(self, fps, fmin=65, fmax=2100, unique_filters=True,
                 models=None, sample_rate=44100, fold=None):
        assert fps <= 10, 'Cannot handle fps larger than 10 yet.'
        assert 10 % fps == 0, 'Needs to be divisible'
        self.factor = int(10 / fps)
        from madmom.audio.chroma import DeepChromaProcessor
        from hashlib import sha1
        import pickle
        self.fps = fps
        self.fmin = fmin
        self.fmax = fmax
        self.unique_filters = unique_filters
        self.dcp = DeepChromaProcessor(
            fmin=fmin, fmax=fmax, unique_filters=unique_filters, models=models
        )
        self.model_hash = sha1(pickle.dumps(self.dcp)).hexdigest()

    @property
    def name(self):
        return 'deepchroma_fps={}_fmin={}_fmax={}_uf={}_mdlhsh={}'.format(
            self.fps, self.fmin, self.fmax, self.unique_filters,
            self.model_hash
        )

    def __call__(self, audio_file):
        return self.dcp(audio_file)[::self.factor].astype(np.float32)


class CnnChordFeatures:

    def __init__(self, fps):
        assert fps <= 10, 'Cannot handle fps larger than 10 yet.'
        assert 10 % fps == 0, 'Needs to be divisible'
        self.factor = int(10 / fps)
        from madmom.features.chords import CNNChordFeatureProcessor
        from hashlib import sha1
        import pickle
        self.fps = fps
        self.cnnp = CNNChordFeatureProcessor()
        self.model_hash = sha1(pickle.dumps(self.cnnp)).hexdigest()

    @property
    def name(self):
        return 'cnnchordfeature_fps={}_mdlhsh={}'.format(
            self.fps, self.model_hash
        )

    def __call__(self, audio_file):
        return self.cnnp(audio_file)[::self.factor].astype(np.float32)


def load_giantsteps_key_dataset(data_dir, feature_cache_dir, feature):

    if feature == 'lfs':
        compute_features = LogFiltSpec(
            frame_sizes=[8192],
            num_bands=24,
            fmin=65,
            fmax=2100,
            fps=5,
            unique_filters=True
        )
    elif feature == 'dc':
        compute_features = DeepChroma(
            fps=5
        )
    elif feature == 'cnn':
        compute_features = CnnChordFeatures(
            fps=5
        )
    else:
        raise ValueError('Invalid feature: {}'.format(feature))

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


def load_billboard_key_dataset(data_dir, feature_cache_dir, feature):
    if feature == 'lfs':
        compute_features = LogFiltSpec(
            frame_sizes=[8192],
            num_bands=24,
            fmin=65,
            fmax=2100,
            fps=5,
            unique_filters=True
        )
    elif feature == 'dc':
        compute_features = DeepChroma(
            fps=5
        )
    elif feature == 'cnn':
        compute_features = CnnChordFeatures(
            fps=5
        )
    else:
        raise ValueError('Invalid feature: {}'.format(feature))

    compute_targets = SingleKeyMajMinTarget()

    return dmgr.Dataset(
        data_dir=data_dir,
        feature_cache_dir=feature_cache_dir,
        split_defs=[join(data_dir, 'splits', f)
                    for f in ['eusipco2017_val.fold', 'eusipco2017_test.fold']],
        source_ext='.flac',
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


def load_giantsteps(data_dir, feature_cache_dir, feature, dist_sampling):

    print('Loading GiantSteps Dataset...')

    test_dataset = load_giantsteps_key_dataset(
        join(data_dir, 'giantsteps-key-dataset'),
        feature_cache_dir,
        feature
    )

    test_set = load_data(
        test_dataset.all_files(),
        use_augmented=True  # no augmented files in there
    )

    print('Loading GiantSteps MTG Dataset...')

    train_dataset = load_giantsteps_key_dataset(
        join(data_dir, 'giantsteps-mtg-key-dataset-augmented'),
        feature_cache_dir,
        feature
    )

    training_files, val_files = train_dataset.random_split([0.8, 0.2])
    training_set = load_data(
        training_files,
        use_augmented=True
    )
    val_set = load_data(
        val_files,
        use_augmented=False
    )

    if dist_sampling:
        l = [np.load(f) for f in training_files['targ'] if '.0.' in f]
        targ_dist = np.bincount(np.hstack(l), minlength=24).astype(np.float)
        targ_dist /= targ_dist.sum()
    else:
        targ_dist = None

    return training_set, val_set, test_set, targ_dist


def load_billboard(data_dir, feature_cache_dir, feature, dist_sampling):

    print('Loading Billboard Dataset..')

    dataset = load_billboard_key_dataset(
        join(data_dir, 'mcgill-billboard-augmented'),
        feature_cache_dir,
        feature
    )

    training_files, val_files, test_files = dataset.fold_split(0, 1)

    training_set = load_data(
        training_files,
        use_augmented=True
    )

    val_set = load_data(
        val_files,
        use_augmented=False
    )

    test_set = load_data(
        test_files,
        use_augmented=False
    )

    print(len(training_set), len(val_set), len(test_set))

    return training_set, val_set, test_set, None

