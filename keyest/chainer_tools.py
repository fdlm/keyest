import random
import Queue
import threading

import chainer
import numpy as np
from chainer.training import extensions


class SequenceIterator(chainer.dataset.Iterator):

    def __init__(self, datasource, batch_size, repeat=True, shuffle=True,
                 augmenters=None):
        self.datasource = datasource
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        self.shuffle = shuffle
        self.offsets = [i * len(datasource) // batch_size
                        for i in range(batch_size)]
        self.augmenters = augmenters or []
        self.epoch_idxs = range(len(datasource))
        if shuffle:
            random.shuffle(self.epoch_idxs)
        self.iteration = 0

    def __next__(self):
        length = len(self.datasource)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration

        idxs, self.epoch_idxs = (self.epoch_idxs[:self.batch_size],
                                 self.epoch_idxs[self.batch_size:])

        data, targets = zip(*[self.datasource[i] for i in idxs])
        data, targets = list(data), list(targets)

        max_len = max(len(d) for d in data)
        for i in range(len(data)):
            if len(data[i]) == max_len:
                continue
            dshape = data[i].shape[1:]
            pad = float(max_len - len(data[i]))
            data[i] = np.vstack([
                np.zeros((int(np.floor(pad / 2)),) + dshape, dtype=np.float32),
                data[i],
                np.zeros((int(np.ceil(pad / 2)),) + dshape, dtype=np.float32)
            ])
            assert len(data[i]) == max_len

        data = np.stack(data)
        targets = np.stack(targets)

        self.iteration += 1
        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            self.epoch_idxs = range(len(self.datasource))
            if self.shuffle:
                random.shuffle(self.epoch_idxs)

        for augment in self.augmenters:
            data, targests = augment(data, targets)

        return zip(data, targets)

    @property
    def epoch_detail(self):
        return float(self.iteration * self.batch_size) / len(self.datasource)

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class ThreadedIterator(chainer.dataset.Iterator):

    def __init__(self, base_iterator, n_cached_items):
        self.base_iterator = base_iterator
        self.queue = Queue.Queue(maxsize=n_cached_items)
        self.end_marker = object()

        def producer():
            for item in self.base_iterator:
                self.queue.put(item)
            self.queue.put(self.end_marker)

        self.thread = threading.Thread(target=producer)
        self.thread.daemon = True
        self.thread.start()

    def __next__(self):
        item = self.queue.get()
        if item is self.end_marker:
            raise StopIteration
        return item

    @property
    def epoch(self):
        return self.base_iterator.epoch

    @property
    def epoch_detail(self):
        return self.base_iterator.epoch_detail

    def serialize(self, serializer):
        self.base_iterator.serialize(serializer)


class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.predictor.set_train(False)
        ret = super(TestModeEvaluator, self).evaluate()
        model.predictor.set_train(True)
        return ret