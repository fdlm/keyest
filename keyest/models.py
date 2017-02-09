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

    def __init__(self, combiner, n_units, device=-1):
        super(BinaryTreeNet, self).__init__(
            combiner=combiner
        )
        self.n_units = n_units
        if device == -1:
            self.pad = np.zeros
        else:
            import cupy
            self.pad = cupy.zeros

    def __call__(self, x):
        batch_size = x.shape[0]
        pad = self.pad((x.shape[0], 1, self.n_units), dtype=np.float32)

        def reshape(x):
            x = F.reshape(x, (batch_size, -1, self.n_units))
            if x.shape[1] % 2 != 0:
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

    def __init__(self, combiner, n_units, device=-1):
        super(BinaryTreeConv, self).__init__(
            combiner=combiner
        )
        self.n_units = n_units
        if device == -1:
            self.pad = np.zeros
        else:
            import cupy
            self.pad = cupy.zeros

    def __call__(self, x):
        x = F.reshape(x, (x.shape[0], 1, x.shape[1], x.shape[2]))
        pad = self.pad((x.shape[0], 1, 1, self.n_units), dtype=np.float32)

        while x.shape[2] != 1:
            if x.shape[2] % 2 != 0:
                x = F.concat([x, pad], axis=2)
            x = self.combiner(x)

        return F.reshape(x, (x.shape[0], self.n_units))


class Mlp(chainer.ChainList):

    def __init__(self,
                 n_layers=3,
                 n_units=256,
                 dropout=0.5,
                 activation=F.relu):
        super(Mlp, self).__init__(
            *[L.Linear(None, n_units) for _ in range(n_layers)]
        )
        self.n_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.train = True

    def set_train(self, train):
        self.train = train

    def __call__(self, x):
        for l in range(self.n_layers):
            x = self[l](x)
            x = F.dropout(x, self.dropout, self.train)
            x = self.activation(x)
        return x


class Mse(chainer.Chain):

    def __init__(self, representation):
        super(Mse, self).__init__(
            representation=representation
        )

    def __call__(self, x, y):
        return F.mean_squared_error(self.representation(x), y)

