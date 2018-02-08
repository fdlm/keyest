from __future__ import print_function

import operator
import pickle
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
from test import test, test_unet


USAGE = """
Usage:
    train.py --data=<S> --model=<S> [--data_params=<S|yaml>] 
             [--model_params=<S|yaml>] [--out=<S>] [--force]
             [--model_history] [--save_pred]

Options:
    --data=<S>  Dataset(s) to load (comma separated if multiple)
    --data_params=<S|yaml>  Data loading parameters in YAML format
    --model=<S>  Model to train
    --model_params=<S|yaml>  Model hyper-parameters in YAML format
    --out=<S>  Experiment output directory (default is model name)
    --force  Overwrite experiment output directory if it already exists.
    --model_history  Save model for each epoch
    --save_pred  Save prediction probabilities when testing
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
        self.model_history = args['--model_history']
        self.save_pred = args['--save_pred']


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

    train_set, val_set, test_set = data.load(datasets=ds.split(','),
                                             **args.data_params)
    Model = models.get_model(args.model)
    train_src, val_src, test_src = data.create_datasources(
        datasets=[train_set, val_set, test_set],
        representations=(Model.source_representations() +
                         Model.target_representations())
    )

    print(colored('\nData:\n', color='blue'))
    print('Train Set:       ', len(train_src))
    print('Validation Set:  ', len(val_src))
    print('Test Set:        ', len(test_src))

    model = Model(feature_shape=train_src.dshape, **args.model_params)
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
    train_batches = model.train_iterator(train_src)
    val_batches = model.test_iterator(val_src)

    val_obs = {
        'loss': model.loss,
        # 'acc': trt.objectives.average_categorical_accuracy
    }
    val_obs.update(model.observables)
    validator = trt.training.Validator(
        model, val_batches, observables=val_obs
    )
    model_checkpoints = trt.outputs.ModelCheckpoint(
        model, file_fmt=join(experiment_dir, 'best_model_ep_{epoch:03d}.pkl'),
        max_history=3)
    checkpoint_on_improvement = trt.training.ImprovementTrigger(
        [model_checkpoints], observed='val_loss')
    callbacks = [checkpoint_on_improvement]
    if args.model_history:
        model.save(join(experiment_dir, 'model_init.pkl'))
        callbacks.append(trt.outputs.ModelCheckpoint(
            model, file_fmt=join(experiment_dir, 'model_ep_{epoch:03d}.pkl')))

    observables = {'loss': model.loss,
                   # 'acc': trt.objectives.average_categorical_accuracy
                   }
    observables.update(model.observables)

    print(colored('Training:\n', color='blue'))
    trt.training.train(
        net=model, train_batches=train_batches,
        num_epochs=model.hypers['n_epochs'],
        observables=observables,
        updater=model.update,
        regularizers=model.regularizers,
        validator=validator,
        logs=[trt.outputs.ConsoleLog(),
              trt.outputs.YamlLog(join(experiment_dir, 'log.yaml'))],
        callbacks=model.callbacks + callbacks,
    )

    # load best parameters
    model.load(model_checkpoints.history[-1])
    os.symlink(basename(model_checkpoints.history[-1]),
               join(experiment_dir, 'best_model.pkl'))
    # create madmom neural network processor if possible
    if hasattr(model, 'to_madmom_processor'):
        pickle.dump(model.to_madmom_processor(),
                    join(experiment_dir, 'best_model_madmom.pkl'),
                    protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------
    # Compute predictions
    # -------------------
    print(colored('\nApplying on Training Set:\n', color='blue'))
    test_unet(model, [t for t in train_src.datasources if '.0' in t.name],
         join(experiment_dir, 'train'), args.save_pred)
    print(colored('\nApplying on Validation Set:\n', color='blue'))
    test_unet(model, val_src.datasources,
         join(experiment_dir, 'val'), args.save_pred)
    print(colored('\nApplying on Test Set:\n', color='blue'))
    test_unet(model, test_src.datasources,
         join(experiment_dir, 'test'), args.save_pred)


if __name__ == '__main__':
    main()
