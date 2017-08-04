from __future__ import print_function

import numpy as np
from os.path import join
import auds
from auds.datasets import Dataset, Views
from auds.representations import SingleKeyMajMin, LogFiltSpec
from trattoria.data import DataSource, AggregatedDataSource
from config import DATASET_DIR, CACHE_DIR


def load(datasets, augmented=True, n_processes=1):
    train_datasets = []
    val_datasets = []
    test_datasets = []
    if 'giantsteps' in datasets:
        test_datasets.append(Dataset.from_directory(
            join(DATASET_DIR, 'giantsteps-key-dataset'),
            views=[Views.audio, Views.key],
            name='GiantSteps Key'
        ))
        gs_train_dir = 'giantsteps-mtg-key-dataset'
        gs_train_name = 'GiantSteps MTG Key'
        if augmented:
            gs_train_dir += '-augmented'
            gs_train_name = ' (augmented)'

        gs_mtg = Dataset.from_directory(
            join(DATASET_DIR, gs_train_dir),
            views=[Views.audio, Views.key],
            name=gs_train_name
        )
        gs_mtg_tr, gs_mtg_va = gs_mtg.random_subsets(
            [0.8, 0.2], np.random.RandomState(4711)
        )
        train_datasets.append(gs_mtg_tr)
        val_datasets.append(gs_mtg_va)
    if 'billboard' in datasets:
        if augmented:
            bb_dir = 'mcgill-billboard-augmented'
            bb_name = 'McGill Billboard (Augmented)'
        else:
            bb_dir = 'mcgill-billboard/unique'
            bb_name = 'McGill Billboard'

        bb = Dataset.from_directory(
            join(DATASET_DIR, bb_dir),
            views=[Views.audio, Views.key],
            name=bb_name
        )

        bb_te = bb.fold_subset('eusipco2017', 'test')
        bb_va = bb.fold_subset('eusipco2017', 'val')
        bb_tr = bb.subtract(bb_te).subtract(bb_va)

        train_datasets.append(bb_tr)
        val_datasets.append(bb_va)
        test_datasets.append(bb_te)

    # TODO: Fix combination of cached and multiprocessed representation
    # src_repr = auds.representations.make_parallel(
    src_repr = auds.representations.make_cached(
    #     auds.representations.make_cached(
            LogFiltSpec(
                frame_size=8192,
                fft_size=None,
                n_bands=24,
                fmin=65,
                fmax=2100,
                fps=5,
                unique_filters=True,
                sample_rate=44100
            ), CACHE_DIR)
        # n_processes=n_processes)
    trg_repr = SingleKeyMajMin()

    train_dataset = sum(train_datasets, Dataset())
    val_dataset = sum(val_datasets, Dataset())
    val_dataset.filter(
        lambda p: not augmented or '.0' in p
    )
    test_dataset = sum(test_datasets, Dataset())

    def create_datasource(dataset):
        sr = src_repr(dataset)
        tr = trg_repr(dataset)
        return AggregatedDataSource(
            [DataSource(data=[sr[p][np.newaxis, ...], tr[p][np.newaxis, ...]],
                        name=p)
             for p in sr])

    train_data = create_datasource(train_dataset)
    val_data = create_datasource(val_dataset)
    test_data = create_datasource(test_dataset)

    return train_data, val_data, test_data

