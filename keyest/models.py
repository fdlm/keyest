import warnings
import theano.tensor as tt

import auds
import lasagne
import theano
import numpy as np
import trattoria
import operator
from abc import abstractmethod, abstractproperty

from auds.representations import LogFiltSpec, SingleKeyMajMin
from config import CACHE_DIR
from lasagne.nonlinearities import softmax, elu
from lasagne.layers import (DenseLayer, InputLayer, dropout_channels,
                            ReshapeLayer, Conv2DLayer, batch_norm, MergeLayer,
                            MaxPool2DLayer, GlobalPoolLayer, NonlinearityLayer,
                            FlattenLayer, DimshuffleLayer, BatchNormLayer,
                            ConcatLayer, ScaleLayer)
from lasagne.init import HeNormal
from trattoria.objectives import average_categorical_crossentropy
from trattoria.nets import NeuralNetwork

try:
    from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
    from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
    from lasagne.layers.dnn import batch_norm_dnn as batch_norm
except ImportError:
    warnings.warn('Cannot import CuDNN implementations!')


class AverageLayer(MergeLayer):

    def __init__(self, incoming, axis, mask_input=None, **kwargs):
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings) - 1
        self.axis = axis
        super(AverageLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[:self.axis] + input_shape[self.axis + 1:]

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]

        if mask is None:
            return tt.mean(input, axis=self.axis)
        else:
            mask = mask.dimshuffle(0, 1, 'x')
            return tt.sum(input * mask, axis=self.axis) / tt.sum(mask, axis=1)


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
        return []

    @staticmethod
    def target_representations():
        return []


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

    def train_iterator(self, datasource):
        return trattoria.iterators.SequenceClassificationIterator(
            datasource.datasources,
            batch_size=8,
            shuffle=True,
            fill_last=True,
        )

    def test_iterator(self, datasource):
        return trattoria.iterators.SequenceClassificationIterator(
            datasource.datasources,
            batch_size=1,
            shuffle=False,
            fill_last=False,
        )

    def to_madmom_procesosor(self, params=None):
        from madmom.ml.nn.layers import (ConvolutionalLayer, FeedForwardLayer,
                                         AverageLayer, PadLayer,
                                         TransposeLayer, ReshapeLayer)
        from madmom.ml.nn.activations import elu, softmax
        from madmom.ml.nn import NeuralNetwork

        layers = []
        p = params or self.get_params()
        for _ in range(self.hypers['n_layers']):
            layers.append(PadLayer(pad=self.hypers['filter_size'] // 2))
            layers.append(ConvolutionalLayer(
                p[0].transpose(1, 0, 2, 3)[:, :, ::-1, ::-1], p[1],
                pad='valid', activation_fn=elu))
            del p[:2]
        layers.append(TransposeLayer((0, 2, 1)))
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


class Snippet(object):

    def __init__(self, snippet_length):
        self.snippet_length = snippet_length

    @abstractmethod
    def snippet_start(self, data, data_length):
        pass

    def __call__(self, batch_iterator):
        for batch in batch_iterator:
            seq, other, mask, target = (batch[0], batch[1:-2], batch[-2],
                                        batch[-1])

            seq_snippet = np.zeros(
                (seq.shape[0], self.snippet_length) + seq.shape[2:],
                dtype=seq.dtype)

            mask_snippet = np.zeros((mask.shape[0], self.snippet_length),
                                    dtype=mask.dtype)

            for i in range(len(seq)):
                dlen = np.flatnonzero(mask[i])[-1]
                start = self.snippet_start(seq[i], dlen)
                end = start + self.snippet_length
                ds = seq[i, start:end, ...]
                ms = mask[i, start:end]
                seq_snippet[i, :len(ds)] = ds
                mask_snippet[i, :len(ms)] = ms

            yield (seq_snippet,) + other + (mask_snippet,) + (target,)


class CenterSnippet(Snippet):

    def snippet_start(self, data, data_length):
        return max(0, data_length / 2 - self.snippet_length / 2)


class RandomSnippet(Snippet):

    def snippet_start(self, data, data_length):
        return np.random.randint(0, max(1, data_length - self.snippet_length))


class BeginningSnippet(Snippet):

    def snippet_start(self, data, data_length):
        return 0


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


class ExpressionMergeLayer(MergeLayer):

    """
    This layer performs an custom expressions on list of inputs to merge them.
    This layer is different from ElemwiseMergeLayer by not required all
    input_shapes are equal

    Parameters
    ----------
    incomings : a list of :class:`Layer` instances or tuples
        the layers feeding into this layer, or expected input shapes

    merge_function : callable
        the merge function to use. Should take two arguments and return the
        updated value. Some possible merge functions are ``theano.tensor``:
        ``mul``, ``add``, ``maximum`` and ``minimum``.

    output_shape : None, callable, tuple, or 'auto'
        Specifies the output shape of this layer. If a tuple, this fixes the
        output shape for any input shape (the tuple can contain None if some
        dimensions may vary). If a callable, it should return the calculated
        output shape given the input shape. If None, the output shape is
        assumed to be the same as the input shape. If 'auto', an attempt will
        be made to automatically infer the correct output shape.

    Notes
    -----
    if ``output_shape=None``, this layer chooses first input_shape as its
    output_shape

    Example
    --------
    >>> from lasagne.layers import InputLayer, DimshuffleLayer, ExpressionMergeLayer
    >>> l_in = lasagne.layers.InputLayer(shape=(None, 500, 120))
    >>> l_mask = lasagne.layers.InputLayer(shape=(None, 500))
    >>> l_dim = lasagne.layers.DimshuffleLayer(l_mask, pattern=(0, 1, 'x'))
    >>> l_out = lasagne.layers.ExpressionMergeLayer(
                                (l_in, l_dim), tensor.mul, output_shape='auto')
    (None, 500, 120)
    """

    def __init__(self, incomings, merge_function, output_shape=None, **kwargs):
        super(ExpressionMergeLayer, self).__init__(incomings, **kwargs)
        if output_shape is None:
            self._output_shape = None
        elif output_shape == 'auto':
            self._output_shape = 'auto'
        elif hasattr(output_shape, '__call__'):
            self.get_output_shape_for = output_shape
        else:
            self._output_shape = tuple(output_shape)

        self.merge_function = merge_function

    def get_output_shape_for(self, input_shapes):
        if self._output_shape is None:
            return input_shapes[0]
        elif self._output_shape is 'auto':
            input_shape = [(0 if s is None else s for s in ishape)
                           for ishape in input_shapes]
            Xs = [T.alloc(0, *ishape) for ishape in input_shape]
            output_shape = self.merge_function(*Xs).shape.eval()
            output_shape = tuple(s if s else None for s in output_shape)
            return output_shape
        else:
            return self._output_shape

    def get_output_for(self, inputs, **kwargs):
        return self.merge_function(*inputs)


class SelecterOutSaver(object):

    def __init__(self, tag_selected_net, batches):
        self.batches = batches
        self.outs = [lasagne.layers.get_output(sel)
                     for sel in tag_selected_net.selecters]
        self.compute_selections = theano.function(
            [tag_selected_net.get_inputs()[1]],
            self.outs,
        )

    def __call__(self, epoch, observed):
        selector_outs = [[] for _ in self.outs]
        for batch in self.batches:
            outs = self.compute_selections(batch[:-1])
            for i in range(len(outs)):
                selector_outs[i].append(outs[i])
        np.save('selectorout.npz', selector_outs)


class TagSelectedSnippet(NeuralNetwork, TrainableModel):

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
                 select_filters=True,
                 tags_at_softmax=False,
                 ):
        # TODO: tags at projection
        # TODO: softmax bias from tags
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
            select_filters=select_filters,
            tags_at_softmax=tags_at_softmax
        )

        self.snippet_length = snippet_length
        self.patience = patience
        self.l2 = l2
        self.learning_rate = theano.shared(np.float32(learning_rate),
                                           allow_downcast=True)

        net = InputLayer((None,) + feature_shape, name='spec in')
        self.bn_alpha = theano.shared(np.float32(0.01))
        tag_input = BatchNormLayer(
            InputLayer((None, 65), name='tag in'),
            beta=None, gamma=lasagne.init.Constant(0.5),
            name='tags_normed', alpha=self.bn_alpha)
        tag_input.params[tag_input.gamma].remove('trainable')
        tag_input = NonlinearityLayer(
            tag_input, nonlinearity=lasagne.nonlinearities.tanh)

        mask = InputLayer((None, None), name='mask in')
        self.mask = mask.input_var
        n_batch, n_time_steps, _ = net.input_var.shape
        net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_shape[-1]))

        for i in range(n_layers):
            net = Conv2DLayer(net, num_filters=n_filters,
                              filter_size=filter_size, pad='same',
                              nonlinearity=elu, W=HeNormal())
            if select_filters:
                selecter = DenseLayer(
                    tag_input, num_units=n_filters,
                    nonlinearity=lasagne.nonlinearities.sigmoid,
                    W=lasagne.init.Constant(0.), b=None)
                selecter = DimshuffleLayer(selecter, (0, 1, 'x', 'x'))
                net = ExpressionMergeLayer([net, selecter], tt.mul)
            if dropout > 0.:
                net = dropout_channels(net, p=dropout)

        net = lasagne.layers.dimshuffle(net, (0, 2, 1, 3))
        net = ReshapeLayer(net, (n_batch * n_time_steps,
                                 n_filters * feature_shape[-1]))
        net = DenseLayer(net, num_units=embedding_size, nonlinearity=elu)
        net = ReshapeLayer(net, (n_batch, n_time_steps, embedding_size))
        net = AverageLayer(net, axis=1, mask_input=mask)
        if tags_at_softmax:
            net = FlattenLayer(net)
            net = ConcatLayer([net, tag_input])
        net = DenseLayer(net, num_units=24, nonlinearity=softmax)

        y = tt.fmatrix('y')
        super(TagSelectedSnippet, self).__init__(net, y)

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
        batch_norm_update = trattoria.schedules.Linear(
            self.bn_alpha, start_epoch=0, end_epoch=2,
            target_value=0.,
        )
        parameter_reset = trattoria.schedules.WarmRestart(
            self, observe='val_acc', patience=self.patience,
            compare=operator.gt)
        return [learn_rate_halving, batch_norm_update, parameter_reset]

    @property
    def observables(self):
        return {'lr': lambda *args: self.learning_rate,
                'bn_alpha': lambda *args: self.bn_alpha}

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

    def test_iterator(self, datasource):
        return trattoria.iterators.SequenceClassificationIterator(
            datasource.datasources,
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
                 n_layers=5,
                 n_filters=8,
                 filter_size=5,
                 dropout=0.0,
                 embedding_size=48,
                 n_epochs=100,
                 learning_rate=0.001,
                 patience=10,
                 l2=1e-4,
                 snippet_length=100):

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
            snippet_length=snippet_length
        )

        self.snippet_length = snippet_length
        self.patience = patience
        self.l2 = l2
        self.learning_rate = theano.shared(np.float32(learning_rate),
                                           allow_downcast=True)

        net = InputLayer((None,) + feature_shape)
        n_batch, n_time_steps, _ = net.input_var.shape
        net = ReshapeLayer(net, (n_batch, 1, n_time_steps, feature_shape[-1]))

        nonlin = lasagne.nonlinearities.elu
        init_conv = lasagne.init.HeNormal
        n_filt = 4

        # --- conv layers ---
        net = Conv2DLayer(net, num_filters=n_filt, filter_size=5, stride=1, pad=2, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = Conv2DLayer(net, num_filters=n_filt, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = MaxPool2DLayer(net, pool_size=2)
        # net = dropout_channels(net, p=0.3)

        net = Conv2DLayer(net, num_filters=n_filt * 2, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = Conv2DLayer(net, num_filters=n_filt * 2, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = MaxPool2DLayer(net, pool_size=2)
        # net = dropout_channels(net, p=0.3)

        net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = Conv2DLayer(net, num_filters=n_filt * 4, filter_size=3, stride=1, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        net = MaxPool2DLayer(net, pool_size=2)
        # net = dropout_channels(net, p=0.3)

        net = Conv2DLayer(net, num_filters=n_filt * 8, filter_size=3, pad=1, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        # net = dropout_channels(net, p=0.5)
        net = Conv2DLayer(net, num_filters=n_filt * 8, filter_size=1, pad=0, W=init_conv(), nonlinearity=nonlin)
        net = batch_norm(net)
        # net = dropout_channels(net, p=0.5)

        # --- feed forward part ---
        net = Conv2DLayer(net, num_filters=24, filter_size=1, W=init_conv(),
                          nonlinearity=nonlin)
        net = batch_norm(net)
        net = GlobalPoolLayer(net)
        net = FlattenLayer(net)
        net = NonlinearityLayer(net, nonlinearity=lasagne.nonlinearities.softmax)
        # net = NonlinearityLayer(net, nonlinearity=temperature_softmax)

        y = tt.fmatrix('y')
        super(AllConv, self).__init__(net, y)

    def update(self, loss, params):
        return lasagne.updates.adam(loss, params, learning_rate=0.001)

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
        it = trattoria.iterators.SequenceClassificationIterator(
            data,
            batch_size=128,
            shuffle=True,
            fill_last=True,
        )
        return trattoria.iterators.AugmentedIterator(
            it, remove_mask, RandomSnippet(snippet_length=self.snippet_length),
        )

    def test_iterator(self, data):
        it = trattoria.iterators.SequenceClassificationIterator(
            data,
            batch_size=128,
            shuffle=False,
            fill_last=False,
        )
        return trattoria.iterators.AugmentedIterator(
            it, remove_mask, CenterSnippet(snippet_length=self.snippet_length * 3),
        )

