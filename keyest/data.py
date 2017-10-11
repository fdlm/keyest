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

        gs_mtg_nonaug = gs_mtg.filter(
            lambda p: not augmented or '.0' in p
        )

        gs_mtg_nonaug_tr, gs_mtg_va = gs_mtg_nonaug.random_subsets(
            [0.8, 0.2], np.random.RandomState(4711)
        )

        def piece_id(p):
            return p.split('.')[0]
        val_piece_ids = set(piece_id(p) for p in gs_mtg_va.pieces.keys())
        gs_mtg_tr = gs_mtg.filter(
            lambda piece: piece_id(piece) not in val_piece_ids
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

        if augmented:
            # do not use pitch shifted songs in validation and test
            bb_te = bb_te.filter(lambda piece: '.0' in piece)
            bb_va = bb_va.filter(lambda piece: '.0' in piece)

        train_datasets.append(bb_tr)
        val_datasets.append(bb_va)
        test_datasets.append(bb_te)
    if 'musicnet' in datasets:
        if augmented:
            mn_dir = 'musicnet-augmented'
            mn_name = 'MusicNet (Augmented)'
        else:
            mn_dir = 'musicnet'
            mn_name = 'MusicNet'

        mn = Dataset.from_directory(
            join(DATASET_DIR, mn_dir),
            views=[Views.audio, Views.key],
            name=mn_name
        )

        mn_te = mn.fold_subset('random_piecewise', 'test')
        mn_va = mn.fold_subset('random_piecewise', 'val')
        mn_tr = mn.subtract(mn_te).subtract(mn_va)

        if augmented:
            # do not use pitch shifted songs in validation and test
            mn_te = mn_te.filter(lambda piece: '.0' in piece)
            mn_va = mn_va.filter(lambda piece: '.0' in piece)

        train_datasets.append(mn_tr)
        val_datasets.append(mn_va)
        test_datasets.append(mn_te)
    if 'cmdb' in datasets:
        if augmented:
            cmdb_dir = 'classical_music_database-augmented'
            cmdb_name = 'Classical Music Database (Augmented)'
        else:
            cmdb_dir = 'classical_music_database'
            cmdb_name = 'Classical Music Database'

        cmdb = Dataset.from_directory(
            join(DATASET_DIR, cmdb_dir),
            views=[Views.audio, Views.key],
            name=cmdb_name
        )

        cmdb_te = cmdb.fold_subset('random', 'test')
        cmdb_va = cmdb.fold_subset('random', 'val')
        cmdb_tr = cmdb.subtract(cmdb_te).subtract(cmdb_va)

        if augmented:
            # do not use pitch shifted songs in validation and test
            cmdb_te = cmdb_te.filter(lambda piece: '.0' in piece)
            cmdb_va = cmdb_va.filter(lambda piece: '.0' in piece)

        train_datasets.append(cmdb_tr)
        val_datasets.append(cmdb_va)
        test_datasets.append(cmdb_te)

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
    # val_dataset = sum(val_datasets, Dataset()).filter(
    #     lambda p: not augmented or '.0' in p
    # )
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

