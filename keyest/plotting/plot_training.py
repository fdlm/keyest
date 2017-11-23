import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from docopt import docopt
from os.path import join
import yaml


USAGE = """
Usage:
    plot_training.py [-c] <exp_name>

Options:
    -c  chainer labels
"""


def main():
    args = docopt(USAGE)
    if args['-c']:
        train_loss = 'main/loss'
        train_acc = 'main/accuracy'
        val_loss = 'validation/main/loss'
        val_acc = 'validation/main/accuracy'
    else:
        train_loss = 'train_loss'
        train_acc = 'train_accuracy'
        val_loss = 'validation_loss'
        val_acc = 'validation_accuracy'

    sns.set(color_codes=True)

    data = pd.DataFrame(yaml.load(open(join(args['<exp_name>'], 'log'))))
    plt.plot(data['epoch'], data[train_loss], label='train loss', c='g')
    plt.plot(data['epoch'], data[val_loss], label='validation loss', c='r')
    plt.plot(data['epoch'], 1. - data[train_acc], label='train error', c='g')
    plt.plot(data['epoch'], 1. - data[val_acc], label='validation error', c='r')
    plt.axhline(min(data[val_loss]), ls='--', c='r')
    plt.axhline(min(1. - data[val_acc]), ls='--', c='r')
    plt.ylim((0, 3.0))
    plt.legend()
    plt.title(args['<exp_name>'])
    plt.show(block=True)


if __name__ == '__main__':
    main()
