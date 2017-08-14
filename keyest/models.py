import warnings
import theano.tensor as tt
import lasagne
import theano
import numpy as np
import trattoria
from functools import partial
from abc import abstractmethod, abstractproperty
from lasagne.nonlinearities import softmax, elu
from lasagne.layers import (DenseLayer, InputLayer, dropout_channels,
                            ReshapeLayer, Conv2DLayer, batch_norm, MergeLayer)
from lasagne.init import HeNormal
from trattoria.objectives import average_categorical_crossentropy
from trattoria.nets import NeuralNetwork

try:
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
    def train_iterator(self, data):
        pass

    @abstractmethod
    def test_iterator(self, data):
        pass


def build_model(model, **kwargs):
    """
    Builds a model class.

    :param model: name of the model (str)
    :param kwargs: hyper-parameters of the model
    :return: model object
    """
    return globals()[model](**kwargs)


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
                 l2=1e-4):

        self._hypers = dict(
            n_layers=n_layers,
            n_filters=n_filters,
            filter_size=filter_size,
            dropout=dropout,
            embedding_size=embedding_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            l2=l2
        )

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
                                                # mask=self.mask)

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
            self.learning_rate, factor=0.5, observe='val_loss', patience=10)
        parameter_reset = trattoria.schedules.WarmRestart(
            self, observe='val_loss', patience=10)
        return [learn_rate_halving, parameter_reset]

    @property
    def observables(self):
        return {'lr': lambda *args: self.learning_rate}

    def train_iterator(self, data):
        return trattoria.iterators.SequenceClassificationIterator(
            data,
            batch_size=8,
            shuffle=True,
            fill_last=True,
        )

    def test_iterator(self, data):
        return trattoria.iterators.SequenceClassificationIterator(
            data,
            batch_size=8,
            shuffle=False,
            fill_last=False,
        )
