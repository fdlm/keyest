import operator
import warnings
from abc import abstractmethod, abstractproperty

import lasagne
import numpy as np
import theano
import theano.tensor as tt
from lasagne.init import HeNormal
from lasagne.layers import (DenseLayer, InputLayer, dropout_channels,
                            ReshapeLayer, Conv2DLayer, batch_norm,
                            MaxPool2DLayer, GlobalPoolLayer, NonlinearityLayer,
                            FlattenLayer, DimshuffleLayer, BatchNormLayer,
                            ConcatLayer, ExpressionLayer)
from lasagne.nonlinearities import softmax, elu
from pastitsio.layers import AverageLayer, ExpressionMergeLayer

import auds
import trattoria
from auds.representations import LogFiltSpec, SingleKeyMajMin
from config import CACHE_DIR
from augmenters import CenterSnippet, RandomSnippet
from trattoria.nets import NeuralNetwork
from trattoria.objectives import average_categorical_crossentropy

try:
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    warnings.warn('Cannot import CuDNN implementations!')


class TrainableModel(object):

    @abstractmethod
    def update(self, loss, params):
        pass

    @abstractmethod
    def loss(self, predictions, targets):
        pass

    @abstractproperty
    def hypers(self):
        pass

    @property
    def regularizers(self):
        return []

    @property
    def callbacks(self):
        return []

    @property
    def observables(self):
        return {}

    @abstractmethod
    def train_iterator(self, datasource):
        pass

    @abstractmethod
    def test_iterator(self, datasource):
        pass

    @staticmethod
    def source_representations():
        raise NotImplementedError('Specify source representation!')

    @staticmethod
    def target_representations():
        raise NotImplementedError('Specify target representation!')


def get_model(model):
    return globals()[model]


def add_dimension(representation):
    import types

    def represent_with_added_dim(self, view_files):
        return self._represent_lowerdim(view_files)[np.newaxis, ...]

    representation._represent_lowerdim = representation.represent
    representation.represent = types.MethodType(represent_with_added_dim,
                                                representation)
    return representation


class Eusipco2017(NeuralNetwork, TrainableModel):

    def __init__(self,
                 feature_shape,
                 n_layers=5,
                 n_filters=8,
                 filter_size=5,
                 dropout=0.0,
                 embedding_size=48,
                 n_epochs=100,
                 learning_rate=0.001,
                 patience=10,
                 l2=1e-4):

        self._hypers = dict(
            n_layers=n_layers,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout=dropout,
            embedding_size=embedding_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            l2=l2,
            patience=patience
        )

        self.patience = patience
        self.l2 = l2
        self.learning_rate = theano.shared(np.float32(learning_rate),
                                           allow_downcast=True)

        net = InputLayer((None,) + feature_shape)
        mask = InputLayer((None, None))
        self.mask = mask.input_var
        n_batch, n_time_steps, _ = net.input_var.shape
        net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_shape[-1]))
        for i in range(n_layers):
            net = Conv2DLayer(net, num_filters=n_filters,
                              filter_size=filter_size, pad='same',
                              nonlinearity=elu, W=HeNormal())
            if dropout > 0.:
                net = dropout_channels(net, p=dropout)

        net = lasagne.layers.dimshuffle(net, (0, 2, 1, 3))
        net = ReshapeLayer(net, (n_batch * n_time_steps,
                                 n_filters * feature_shape[-1]))
        net = DenseLayer(net, num_units=embedding_size, nonlinearity=elu)
        net = ReshapeLayer(net, (n_batch, n_time_steps, embedding_size))
        net = AverageLayer(net, axis=1, mask_input=mask)
        net = DenseLayer(net, num_units=24, nonlinearity=softmax)

        y = tt.fmatrix('y')
        super(Eusipco2017, self).__init__(net, y)

    def update(self, loss, params):
        return lasagne.updates.momentum(loss, params,
                                        learning_rate=self.learning_rate,
                                        momentum=0.9)

    def loss(self, predictions, targets):
        return average_categorical_crossentropy(predictions, targets)

    @property
    def hypers(self):
        return self._hypers

    @property
    def regularizers(self):
        return [lasagne.regularization.regularize_layer_params(
            self.net, lasagne.regularization.l2) * self.l2]

    @property
    def callbacks(self):
        learn_rate_halving = trattoria.schedules.PatienceMult(
            self.learning_rate, factor=0.5, observe='val_acc',
            patience=self.patience, compare=operator.gt)
        parameter_reset = trattoria.schedules.WarmRestart(
            self, observe='val_acc', patience=self.patience,
            compare=operator.gt)
        return [learn_rate_halving, parameter_reset]

    @property
    def observables(self):
        return {'lr': lambda *args: self.learning_rate}

    def train_iterator(self, data):
        return trattoria.iterators.SequenceClassificationIterator(
            data.datasources,
            batch_size=8,
            shuffle=True,
            fill_last=True,
        )

    def test_iterator(self, data):
        return trattoria.iterators.SequenceClassificationIterator(
            data.datasources,
            batch_size=1,
            shuffle=False,
            fill_last=False,
        )

    def to_madmom_nn(self):
        from madmom.ml.nn.layers import (ConvolutionalLayer, FeedForwardLayer,
                                         TransposeLayer, ReshapeLayer,
                                         AverageLayer, PadLayer)
        from madmom.ml.nn.activations import elu, softmax
        from madmom.ml.nn import NeuralNetwork

        layers = []
        p = self.get_param_values()
        for _ in range(self.hypers['n_layers']):
            layers.append(PadLayer(width=self.hypers['filter_size'] // 2,
                                   axes=(0, 1)))
            layers.append(ConvolutionalLayer(
                # this is necessary if filters are flipped
                p[0].transpose(1, 0, 2, 3)[:, :, ::-1, ::-1], p[1],
                pad='valid', activation_fn=elu))
            del p[:2]
        layers.append(TransposeLayer(axes=(0, 2, 1)))
        layers.append(ReshapeLayer((-1, 105 * self.hypers['n_filters'])))
        layers.append(FeedForwardLayer(p[0], p[1], activation_fn=elu))
        del p[:2]
        layers.append(AverageLayer(axis=0, keepdims=True))
        layers.append(FeedForwardLayer(p[0], p[1], activation_fn=softmax))
        return NeuralNetwork(layers)

    @staticmethod
    def source_representations():
        return [add_dimension(auds.representations.make_cached(
            LogFiltSpec(frame_size=8192, fft_size=None, n_bands=24,
                        fmin=65, fmax=2100, fps=5, unique_filters=True,
                        sample_rate=44100), CACHE_DIR))]

    @staticmethod
    def target_representations():
        return [SingleKeyMajMin()]


class Eusipco2017Snippet(Eusipco2017):

    def __init__(self, snippet_length=100, **kwargs):
        super(Eusipco2017Snippet, self).__init__(**kwargs)
        self._hypers['snippet_length'] = snippet_length
        self.snippet_length = snippet_length

    def train_iterator(self, data):
        return trattoria.iterators.SubsetIterator(
            trattoria.iterators.AugmentedIterator(
                super(Eusipco2017Snippet, self).train_iterator(data),
                RandomSnippet(snippet_length=self.snippet_length)
            ),
            percentage=0.25
        )


class TagSelectedSnippet(NeuralNetwork, TrainableModel):
    """
    Parameters
    ----------
    tags_select_filters : bool
        Use Tag input to compute a weight vector to be multiplied on the
        feature maps. This can be seen as activating/deactivating certain
        feature maps given the tag information.

    tags_at_softmax : bool
        Tags get concatenated to the penultimate representation, before the
        last projection. This is equivalent to computing a key bias given
        the tags.

    tags_at_projection : bool
        Tags get concatenated to each time step before the projection layer
        after the convolutions. This can be seen as computing a projection bias
        given the tags.


    """

    def __init__(self,
                 feature_shape,
                 n_layers=5,
                 n_filters=8,
                 filter_size=5,
                 dropout=0.0,
                 embedding_size=48,
                 n_epochs=100,
                 learning_rate=0.001,
                 patience=10,
                 l2=1e-4,
                 snippet_length=100,
                 tags_select_filters=True,
                 tags_at_softmax=False,
                 tags_at_projection=False,
                 tags_for_softmax_weights=False
                 ):

        # TODO: softmax weights from tags
        # TODO: projection weights from tags
        # TODO: convolutional kernels from tags?

        self._hypers = dict(
            n_layers=n_layers,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout=dropout,
            embedding_size=embedding_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            l2=l2,
            patience=patience,
            snippet_length=snippet_length,
            tags_select_filters=tags_select_filters,
            tags_at_softmax=tags_at_softmax,
            tags_at_projection=tags_at_projection
        )

        self.snippet_length = snippet_length
        self.patience = patience
        self.l2 = l2
        self.learning_rate = theano.shared(np.float32(learning_rate),
                                           allow_downcast=True)

        net = InputLayer((None,) + feature_shape, name='spec in')
        self.spec_in = net.input_var
        tag_inlayer = InputLayer((None, 65), name='tag in')
        self.tag_in = tag_inlayer.input_var
        tag_input = BatchNormLayer(
            tag_inlayer,
            beta=None, gamma=lasagne.init.Constant(0.5),
            name='tags_normed', alpha=0.001)
        tag_input.params[tag_input.gamma].remove('trainable')
        tag_input = NonlinearityLayer(
            tag_input, nonlinearity=lasagne.nonlinearities.tanh)
        self.tag_input = tag_input

        mask = InputLayer((None, None), name='mask in')
        self.mask_in = mask.input_var
        n_batch, n_time_steps, _ = net.input_var.shape
        net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_shape[-1]))

        for i in range(n_layers):
            net = Conv2DLayer(net, num_filters=n_filters,
                              filter_size=filter_size, pad='same',
                              nonlinearity=elu, W=HeNormal())
            if tags_select_filters:
                selecter = DenseLayer(
                    tag_input, num_units=n_filters,
                    nonlinearity=lasagne.nonlinearities.sigmoid,
                    W=lasagne.init.Constant(0.), b=None)
                selecter = DimshuffleLayer(selecter, (0, 1, 'x', 'x'))
                net = ExpressionMergeLayer([net, selecter], tt.mul)
            if dropout > 0.:
                net = dropout_channels(net, p=dropout)

        net = lasagne.layers.dimshuffle(net, (0, 2, 1, 3))

        if tags_at_projection:
            tag_proj = DimshuffleLayer(tag_input, (0, 'x', 1))

            # TODO: check if this really does what I think it does (repeat
            #       the tags for n_time_steps!)
            tag_proj = ExpressionLayer(
                tag_proj, lambda x: theano.tensor.tile(x, (1, n_time_steps, 1)),
                output_shape=lambda in_shape: (in_shape[0], None, in_shape[2])
            )

            net = FlattenLayer(net, outdim=3)
            net = ConcatLayer([net, tag_proj], axis=2)
            net = ReshapeLayer(net, (n_batch * n_time_steps,
                                     n_filters * feature_shape[-1] + 65))
        else:
            net = ReshapeLayer(net, (n_batch * n_time_steps,
                                     n_filters * feature_shape[-1]))
        net = DenseLayer(net, num_units=embedding_size, nonlinearity=elu,
                         name='projection')
        net = ReshapeLayer(net, (n_batch, n_time_steps, embedding_size))
        net = AverageLayer(net, axis=1, mask_input=mask)
        self.avg = net
        if tags_at_softmax:
            net = ConcatLayer([net, tag_input])

        if tags_for_softmax_weights:
            in_units = net.output_shape[1]
            sm_weight_layer = DenseLayer(tag_input, num_units=in_units * 24)
            sm_weight_layer = ReshapeLayer(sm_weight_layer, (in_units, 24))
            sm_weights = lasagne.layers.get_output(sm_weight_layer)
        else:
            sm_weights = lasagne.init.Constant(0.)

        net = DenseLayer(net, num_units=24, nonlinearity=softmax, W=sm_weights)

        y = tt.fmatrix('y')
        super(TagSelectedSnippet, self).__init__(net, y)

    def get_inputs(self):
        return [self.spec_in, self.tag_in, self.mask_in]

    def update(self, loss, params):
        return lasagne.updates.adam(loss, params,
                                    learning_rate=self.learning_rate)

    def loss(self, predictions, targets):
        return average_categorical_crossentropy(predictions, targets, eta=1e-5)

    @property
    def hypers(self):
        return self._hypers

    @property
    def regularizers(self):
        return [lasagne.regularization.regularize_layer_params(
            self.net, lasagne.regularization.l2) * self.l2]

    @property
    def callbacks(self):
        return [trattoria.schedules.Linear(
            self.learning_rate, 10, 100, 0.0
        )]
        # learn_rate_halving = trattoria.schedules.PatienceMult(
        #     slf.learning_rate, factor=0.5, observe='val_acc',
        #     patience=self.patience, compare=operator.gt)
        # parameter_reset = trattoria.schedules.WarmRestart(
        #     self, observe='val_acc', patience=self.patience,
        #     compare=operator.gt)
        # return [learn_rate_halving, parameter_reset]

    @property
    def observables(self):
        return {'lr': lambda *args: self.learning_rate}

    def train_iterator(self, data):
        return trattoria.iterators.SubsetIterator(
            trattoria.iterators.AugmentedIterator(
                trattoria.iterators.SequenceClassificationIterator(
                    data.datasources,
                    batch_size=8,
                    shuffle=True,
                    fill_last=True,
                ),
                RandomSnippet(snippet_length=self.snippet_length)
            ),
            percentage=0.25
        )

    def test_iterator(self, data):
        return trattoria.iterators.SequenceClassificationIterator(
            data.datasources,
            batch_size=1,
            shuffle=False,
            fill_last=False,
        )

    @staticmethod
    def source_representations():
        spectrogram = add_dimension(auds.representations.make_cached(
            LogFiltSpec(frame_size=8192, fft_size=None, n_bands=24,
                        fmin=65, fmax=2100, fps=5, unique_filters=True,
                        sample_rate=44100), CACHE_DIR))
        tags = add_dimension(auds.representations.Precomputed(
            '/home/filip/.tmp/jamendo_tags', 'audio', 'jamendo_tags'))
        return [spectrogram, tags]

    @staticmethod
    def target_representations():
        return [auds.representations.make_cached(SingleKeyMajMin(), CACHE_DIR)]


def remove_mask(batch_iterator):
    for d, m, t in batch_iterator:
        yield d, t


class AllConv(NeuralNetwork, TrainableModel):

    def __init__(self,
                 feature_shape,
                 n_filters=2,
                 dropout=0.0,
                 n_epochs=100,
                 learning_rate=0.001,
                 patience=10,
                 l2=1e-6,
                 snippet_length=100):

        self._hypers = dict(
            n_filters=2,
            dropout=dropout,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            patience=patience,
            snippet_length=snippet_length
        )

        self.snippet_length = snippet_length
        self.patience = patience
        self.l2 = l2
        self.learning_rate = theano.shared(np.float32(learning_rate),
                                           allow_downcast=True)

        nonlin = lasagne.nonlinearities.elu
        init_conv = lasagne.init.HeNormal

        net = InputLayer((None,) + feature_shape)
        n_batch, n_time_steps, _ = net.input_var.shape
        net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_shape[-1]))

        def conv_bn(net, n_filters, filter_size):
            return batch_norm(Conv2DLayer(
                net, num_filters=n_filters, filter_size=filter_size,
                stride=1, pad='same', W=init_conv(), nonlinearity=nonlin))

        # --- conv layers ---
        net = conv_bn(net, n_filters, 5)
        net = conv_bn(net, n_filters, 3)
        net = MaxPool2DLayer(net, pool_size=2)
        if dropout:
            net = dropout_channels(net, p=dropout)

        net = conv_bn(net, n_filters * 2, 3)
        net = conv_bn(net, n_filters * 2, 3)
        net = MaxPool2DLayer(net, pool_size=2)
        if dropout:
            net = dropout_channels(net, p=dropout)

        net = conv_bn(net, n_filters * 4, 3)
        net = conv_bn(net, n_filters * 4, 3)
        net = MaxPool2DLayer(net, pool_size=2)
        if dropout:
            net = dropout_channels(net, p=dropout)

        net = conv_bn(net, n_filters * 8, 3)
        if dropout:
            net = dropout_channels(net, p=dropout)
        net = conv_bn(net, n_filters * 8, 3)
        if dropout:
            net = dropout_channels(net, p=dropout)

        # --- feed forward part ---
        net = conv_bn(net, 24, 1)
        net = GlobalPoolLayer(net)
        net = FlattenLayer(net)
        net = NonlinearityLayer(net, lasagne.nonlinearities.softmax)

        y = tt.fmatrix('y')
        super(AllConv, self).__init__(net, y)

    def update(self, loss, params):
        return lasagne.updates.adam(loss, params,
                                    learning_rate=self.learning_rate)

    def loss(self, predictions, targets):
        return average_categorical_crossentropy(predictions, targets)

    @property
    def hypers(self):
        return self._hypers

    @property
    def regularizers(self):
        return [lasagne.regularization.regularize_layer_params(
            self.net, lasagne.regularization.l2) * self.l2]

    @property
    def callbacks(self):
        return [trattoria.schedules.Linear(
            self.learning_rate, 10, self.hypers['n_epochs'], 0.0
        )]

    @property
    def observables(self):
        return {'lr': lambda *args: self.learning_rate}

    def train_iterator(self, data):
        return trattoria.iterators.SubsetIterator(
            trattoria.iterators.AugmentedIterator(
                trattoria.iterators.SequenceClassificationIterator(
                    data.datasources,
                    batch_size=32,
                    shuffle=True,
                    fill_last=True
                ),
                remove_mask, RandomSnippet(snippet_length=self.snippet_length)
            ),
            percentage=0.25
        )

    def test_iterator(self, data):
        return trattoria.iterators.AugmentedIterator(
            trattoria.iterators.SequenceClassificationIterator(
                data.datasources,
                batch_size=1,
                shuffle=False,
                fill_last=False,
            ),
            remove_mask
        )

    @staticmethod
    def source_representations():
        return [add_dimension(auds.representations.make_cached(
            LogFiltSpec(frame_size=8192, fft_size=None, n_bands=24,
                        fmin=65, fmax=2100, fps=5, unique_filters=True,
                        sample_rate=44100), CACHE_DIR))]

    @staticmethod
    def target_representations():
        return [SingleKeyMajMin()]
