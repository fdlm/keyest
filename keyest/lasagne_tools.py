import numpy as np
import random


class SequenceIterator(object):

    def __init__(self, datasource, batch_size, shuffle=True, augmenters=None,
                 distribution=None):
        self.datasource = datasource
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmenters = augmenters or []
        if distribution is None:
            self.distribution = None
        else:
            _, targets = zip(*datasource)
            true_dist = np.bincount(targets, minlength=24)
            target_prob = distribution / true_dist
            self.distribution = np.array([target_prob[t] for t in targets])

        self.epoch_idxs = []

    def _select_idxs(self):
        if not self.shuffle:
            self.epoch_idxs = range(len(self.datasource))
        else:
            if self.distribution is None:
                self.epoch_idxs = range(len(self.datasource))
                random.shuffle(self.epoch_idxs)
            else:
                self.epoch_idxs = np.random.choice(
                    len(self.datasource),
                    len(self.datasource),
                    p=self.distribution
                )

    def __iter__(self):
        self._select_idxs()
        return self

    def __next__(self):
        if len(self.epoch_idxs) == 0:
            raise StopIteration

        idxs, self.epoch_idxs = (self.epoch_idxs[:self.batch_size],
                                 self.epoch_idxs[self.batch_size:])

        data, targets = zip(*[self.datasource[i] for i in idxs])
        data, targets = list(data), list(targets)
        masks = []

        max_len = max(len(d) for d in data)
        for i in range(len(data)):
            if len(data[i]) == max_len:
                masks.append(np.ones(len(data[i]), dtype=np.float32))
                continue
            dshape = data[i].shape[1:]
            pad = float(max_len - len(data[i]))
            start_pad = int(np.floor(pad / 2))
            end_pad = int(np.ceil(pad / 2))
            data[i] = np.vstack([
                np.zeros((start_pad,) + dshape, dtype=np.float32),
                data[i],
                np.zeros((end_pad,) + dshape, dtype=np.float32)
            ])
            assert len(data[i]) == max_len
            mask = np.ones(len(data[i]), dtype=np.float32)
            mask[:start_pad] = 0.
            mask[-end_pad] = 0.
            masks.append(mask)

        data = np.stack(data)
        targets = np.stack(targets)
        masks = np.stack(masks)

        for augment in self.augmenters:
            data, targets = augment(data, targets)

        return zip(data, targets, masks)

    def next(self):
        return self.__next__()

    @property
    def n_elements(self):
        return len(self.datasource) // self.batch_size + 1
