from docopt import docopt
from glob import glob
from madmom.ml.nn import NeuralNetwork
from madmom.ml.nn.layers import BatchNormLayer, FeedForwardLayer
from madmom.ml.nn.activations import sigmoid, tanh
from tqdm import tqdm
from os.path import join
import os
import numpy as np
import matplotlib
matplotlib.use('cairo')
import matplotlib.pyplot as plt


USAGE = """
Plots which feature maps have been selected by the tag informed net.

Usage:
    plot_selections.py [options] <exp_dir> <tag_dir>
    
Arguments:
    <exp_dir>  Directory of experiment
    <tag_dir>  Directory containing the tags

Options:
    --out=<S>  Directory where to put the figures [default: ./figs]
    --cm  Plot CMDB
    --max_pieces=<I>  Max number of pieces per dataset to use [default: 10]
    --tag_emb  Plot tag embeddings
    --bn_params  Plot batch norm params
    --sel  Plot selections
"""


def identity(x):
    return x


def main():
    args = docopt(USAGE)
    use_cm = args['--cm']
    plot_tag_emb = args['--tag_emb']
    plot_bn = args['--bn_params']
    plot_sel = args['--sel']
    max_pieces = int(args['--max_pieces'])
    model_params = sorted(glob(args['<exp_dir>'] + '/model_ep*.pkl'))
    out_dir = args['--out']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if plot_bn:
        mean = []
        std = []

        for param_file in model_params:
            params = np.load(param_file)
            mean.append(params[3])
            std.append(1. / params[4])

        mean = np.array(mean)
        std = np.array(std)

        fig, axes = plt.subplots(1, 2, sharex=True)
        axes[0].plot(mean, c='r')
        axes[0].set_title('mean')
        axes[1].plot(std, c='b')
        axes[1].set_title('std')
        fig.suptitle('Batch Norm Params')
        fig.savefig(join(out_dir, 'bn_params.pdf'))

    gs_tags = np.array([
        np.load(f)
        for f in sorted(glob(args['<tag_dir>'] + '/*LO*.0.npy'))[:max_pieces]])
    bb_tags = np.array(
        [np.load(f)
         for f in sorted(glob(args['<tag_dir>'] + '/mc*.0.npy'))[:max_pieces]])

    n_pieces = len(gs_tags) + len(bb_tags)

    if use_cm:
        cm_tags = np.array([
            np.load(f)
            for f in sorted(glob(args['<tag_dir>'] + '/cm*.0.npy'))[:max_pieces]])
        n_pieces += len(cm_tags)

    alpha = 5. / n_pieces

    if plot_tag_emb:
        plt.figure(figsize=(15, 10))
        plt.plot(gs_tags.T, c='r', alpha=alpha)
        plt.plot(bb_tags.T, c='b', alpha=alpha)
        if use_cm:
            plt.plot(cm_tags.T, c='g', alpha=alpha)
        plt.xlim(0, 64)
        plt.ylim(-6, 6)
        plt.title('Tag Embeddings, no BN')
        plt.savefig(join(out_dir, 'tag_embeddings.png'))

    for i, param_file in tqdm(enumerate(model_params)):
        params = np.load(param_file)

        bn = BatchNormLayer(beta=0, gamma=params[2],
                            mean=params[3], inv_std=params[4],
                            activation_fn=tanh)
        if plot_tag_emb:
            fig = plt.figure(figsize=(15, 10))
            plt.plot(bn.activate(gs_tags).T, c='r', alpha=alpha)
            plt.plot(bn.activate(bb_tags).T, c='b', alpha=alpha)
            if use_cm:
                plt.plot(bn.activate(cm_tags).T, c='g', alpha=alpha)
            plt.xlim(0, 64)
            plt.ylim(-1, 1)
            plt.title('Tag Embeddings, Batch Norm Ep. {:03d}'.format(i + 1))
            plt.savefig(join(out_dir,
                             'tag_embeddings_ep{:03d}.png'.format(i + 1)))
            del fig

        if plot_sel:
            proj = [
                NeuralNetwork([
                    bn, FeedForwardLayer(weights=w, bias=np.array(0.),
                                         activation_fn=sigmoid)
                ])
                for w in params[5:-2:3]
            ]
            fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True,
                                     sharey=True)
            from itertools import chain
            axes = chain.from_iterable(axes)
            for l, (p, ax) in enumerate(zip(proj, axes)):
                ax.plot(p(gs_tags).T, c='r', alpha=alpha)
                ax.plot(p(bb_tags).T, c='b', alpha=alpha)
                if use_cm:
                    ax.plot(p(cm_tags).T, c='g', alpha=alpha)
                ax.set_title('Layer {}'.format(l))
                ax.set_xlim(0, 23)
                ax.set_ylim(0, 1)

            fig.suptitle('Fmap selection Ep. {:03d}'.format(i + 1))
            fig.savefig(join(out_dir, 'selections_ep{:03d}.png'.format(i + 1)))
            del fig


if __name__ == "__main__":
    main()
