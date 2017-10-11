from __future__ import print_function

import operator
import os
import shutil
from os.path import join, basename

import yaml
from docopt import docopt
from termcolor import colored

import data
import models
import trattoria as trt
from config import EXPERIMENT_ROOT
from test import test


USAGE = """
Usage:
    train.py --data=<S> --model=<S> [--data_params=<S|yaml>] 
             [--model_params=<S|yaml>] [--out=<S>] [--force]

Options:
    --data=<S>  Dataset(s) to load (comma separated if multiple)
    --data_params=<S|yaml>  Data loading parameters in YAML format
    --model=<S>  Model to train
    --model_params=<S|yaml>  Model hyper-parameters in YAML format
    --out=<S>  Experiment output directory (default is model name)
    --force  Overwrite experiment output directory if it already exists.
"""


class Args(object):
    def __init__(self):
        args = docopt(USAGE)
        self.data = args['--data']
        self.data_params = yaml.load(args['--data_params'] or '{}')
        self.model = args['--model']
        self.model_params = yaml.load(args['--model_params'] or '{}')
        self.out = args['--out']
        self.force = args['--force']


def main():
    args = Args()

    # ---------------------------
    # Create experiment directory
    # ---------------------------
    out_dir_name = args.out or args.model
    experiment_dir = join(EXPERIMENT_ROOT, out_dir_name)
    if os.path.exists(experiment_dir):
        if not args.force:
            print('ERROR: experiment directory already exists!')
            return 1
        else:
            shutil.rmtree(experiment_dir)
    os.makedirs(experiment_dir)

    # -------------------------
    # Load data and build model
    # -------------------------
    if args.data == 'all':
        ds = 'giantsteps,billboard,musicnet,cmdb'
    else:
        ds = args.data
    train_set, val_set, test_set = data.load(
        datasets=ds,
        **args.data_params
    )
    print(colored('\nData:\n', color='blue'))
    print('Train Set:       ', len(train_set))
    print('Validation Set:  ', len(val_set))
    print('Test Set:        ', len(test_set))

    model = models.build_model(args.model,
                               feature_shape=train_set.dshape,
                               **args.model_params)
    print(colored('\nModel:\n', color='blue'))
    print(model)
    print('\n')

    yaml.dump({'model': args.model,
               'model_params': model.hypers,
               'data': args.data.split(','),
               'data_params': args.data_params,
               },
              open(join(experiment_dir, 'config.yaml'), 'w'))

    # -----------
    # Train Model
    # -----------
    train_batches = model.train_iterator(train_set.datasources)
    val_batches = model.test_iterator(val_set.datasources)

    validator = trt.training.Validator(
        model, val_batches,
        observables={
            'loss': model.loss,
            'acc': trt.objectives.average_categorical_accuracy
        }
    )
    model_checkpoints = trt.outputs.ModelCheckpoint(
        model,
        file_fmt=join(experiment_dir, 'model_ep_{epoch:03d}.pkl'),
        max_history=3)
    checkpoint_on_improvement = trt.training.ImprovementTrigger(
        [model_checkpoints],
        observed='val_acc',
        compare=operator.gt)

    print(colored('Training:\n', color='blue'))

    trt.training.train(
        net=model, train_batches=train_batches,
        num_epochs=model.hypers['n_epochs'],
        observables=dict({'loss': model.loss,
                          'acc': trt.objectives.average_categorical_accuracy},
                         **model.observables),
        updater=model.update,
        regularizers=model.regularizers,
        validator=validator,
        logs=[trt.outputs.ConsoleLog(),
              trt.outputs.YamlLog(join(experiment_dir, 'log.yaml'))],
        callbacks=model.callbacks + [checkpoint_on_improvement],
    )

    # load best parameters
    model.load(model_checkpoints.history[-1])
    os.symlink(basename(model_checkpoints.history[-1]),
               join(experiment_dir, 'best_model.pkl'))

    # -------------------
    # Compute predictions
    # -------------------
    print(colored('\nApplying on Test Set:\n', color='blue'))
    test(model, test_set, experiment_dir)


if __name__ == '__main__':
    main()
