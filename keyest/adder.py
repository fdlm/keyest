import numpy as np
from models import (DenseCombiner, BinaryTreeNet, ConvCombiner,
                    BinaryTreeConv, Mse)
import chainer
import chainer.functions as F
from chainer.training import extensions


N = 4
N_DATA = 1000
SEQ_LEN = 50
CONV = True


# "load" data
np.random.seed(4711)
x = np.random.random((N_DATA, SEQ_LEN, N)).astype(np.float32)
y = x.sum(axis=1)
train_set = zip(x, y)

# create net and trainer
if not CONV:
    net = Mse(BinaryTreeNet(DenseCombiner(N, F.identity), N))
else:
    net = Mse(BinaryTreeConv(ConvCombiner(N, F.identity), N))

opt = chainer.optimizers.RMSpropGraves()
opt.setup(net)
opt.add_hook(chainer.optimizer.GradientClipping(10.0))
train_it = chainer.iterators.SerialIterator(train_set, batch_size=10)
upd = chainer.training.updater.StandardUpdater(train_it, opt)
trainer = chainer.training.Trainer(upd, stop_trigger=(10, 'epoch'))

# add fancy stuff to trainer
trainer.extend(extensions.ProgressBar())

print net(x, y).data
trainer.run()
print net(x, y).data
