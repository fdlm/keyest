import numpy as np
from models import (DenseCombiner, BinaryTreeNet, ConvCombiner,
                    BinaryTreeConv, Mse)
import chainer
import chainer.functions as F
from chainer.training import extensions


N = 4
N_DATA = 1000
SEQ_LEN = 500
CONV = True


# "load" data
np.random.seed(4711)
x = np.random.random((N_DATA, SEQ_LEN, N)).astype(np.float32)
y = x.sum(axis=1)
train_set = zip(x, y)

if chainer.cuda.available:
    device = 0
else:
    device = -1

# create net and trainer
if not CONV:
    net = Mse(BinaryTreeNet(DenseCombiner(N, F.identity), N, device))
else:
    net = Mse(BinaryTreeConv(ConvCombiner(N, F.identity), N, device))

if device == 0:
    chainer.cuda.get_device(device).use()
    net.to_gpu()
    x = chainer.cuda.to_gpu(x, device=device)
    y = chainer.cuda.to_gpu(y, device=device)

opt = chainer.optimizers.RMSpropGraves()
opt.setup(net)
opt.add_hook(chainer.optimizer.GradientClipping(10.0))
train_it = chainer.iterators.SerialIterator(train_set, batch_size=1,
                                            shuffle=False)
upd = chainer.training.updater.StandardUpdater(train_it, opt, device=device)
trainer = chainer.training.Trainer(upd, stop_trigger=(10, 'epoch'))

# add fancy stuff to trainer
trainer.extend(extensions.ProgressBar())

out = net(x, y)
out.to_cpu()
print out.data
trainer.run()
out = net(x, y)
out.to_cpu()
print out.data
