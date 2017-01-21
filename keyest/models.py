import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F


class DenseCombiner(chainer.Chain):

    def __init__(self, n_units, activation):
        super(DenseCombiner, self).__init__(
            f=L.Linear(2 * n_units, n_units)
        )
        self.activation = activation

    def __call__(self, x):
        return self.activation(self.f(x))


class BinaryTreeNet(chainer.Chain):

    def __init__(self, combiner, n_units):
        super(BinaryTreeNet, self).__init__(
            combiner=combiner
        )
        self.n_units = n_units

    def __call__(self, x):
        batch_size = x.shape[0]

        def reshape(x):
            x = F.reshape(x, (batch_size, -1, self.n_units))
            if x.shape[1] % 2 != 0:
                pad = np.zeros((x.shape[0], 1, self.n_units), dtype=np.float32)
                x = F.concat([x, pad], axis=1)
            return F.reshape(x, (-1, 2, self.n_units))

        pred = reshape(x)
        while pred.shape[0] != batch_size:
            pred = self.combiner(pred)
            pred = reshape(pred)

        return self.combiner(pred)


class ConvCombiner(chainer.Chain):

    def __init__(self, n_units, activation):
        super(ConvCombiner, self).__init__(
            f=L.Convolution2D(1, n_units,
                              ksize=(2, n_units), stride=(2, 1))
        )
        self.activation = activation

    def __call__(self, x):
        return F.transpose(self.activation(self.f(x)), (0, 3, 2, 1))


class BinaryTreeConv(chainer.Chain):

    def __init__(self, combiner, n_units):
        super(BinaryTreeConv, self).__init__(
            combiner=combiner
        )
        self.n_units = n_units

    def __call__(self, x):
        x = F.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        pad = np.zeros((x.shape[0], 1, 1, self.n_units), dtype=np.float32)

        while x.shape[2] != 1:
            if x.shape[2] % 2 != 0:
                x = F.concat([x, pad], axis=2)
            x = self.combiner(x)

        return F.reshape(x, (x.shape[0], self.n_units))


class Mse(chainer.Chain):

    def __init__(self, representation):
        super(Mse, self).__init__(
            representation=representation
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.representation(x), y)
