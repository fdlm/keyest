from abc import abstractmethod

import numpy as np
import random
from scipy.ndimage import shift


class SemitoneShift(object):

    def __init__(self, p, max_shift, bins_per_semitone):
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, data, targets):
        batch_size = len(data)
        shifts = np.random.randint(-self.max_shift, self.max_shift + 1,
                                   batch_size)

        # zero out shifts for 1-p percentage
        no_shift = random.sample(range(batch_size),
                                 int(batch_size * (1 - self.p)))
        shifts[no_shift] = 0

        target_root = (targets % 12) + shifts
        target_type = targets / 12
        new_targets = target_root + target_type * 12

        new_data = np.empty_like(data)
        length = new_data.shape[1]
        for i in range(batch_size):
            # new_data[i] = np.roll(
            #     data[i], shifts[i] * self.bins_per_semitone, axis=-1)
            shift = shifts[i] * self.bins_per_semitone
            pad = np.zeros((length, abs(shift)))
            if shift < 0:
                new_data[i] = np.hstack([data[i, :, abs(shift):], pad])
            elif shift > 0:
                new_data[i] = np.hstack([pad, data[i, :, :-abs(shift)]])
            else:
                new_data[i] = data[i]

        return new_data, new_targets


class Detuning(object):

    def __init__(self, p, max_shift, bins_per_semitone):
        if max_shift >= 0.5:
            raise ValueError('Detuning only works up to half a semitone!')
        self.p = p
        self.max_shift = max_shift
        self.bins_per_semitone = bins_per_semitone

    def __call__(self, data, targets):
        batch_size = len(data)

        shifts = np.random.rand(batch_size) * 2 * self.max_shift - \
                 self.max_shift

        # zero out shifts for 1-p percentage
        no_shift = random.sample(range(batch_size),
                                 int(batch_size * (1 - self.p)))
        shifts[no_shift] = 0

        new_data = np.empty_like(data)
        for i in range(batch_size):
            if shifts[i] == 0:
                new_data[i] = data[i]
            else:
                new_data[i] = shift(
                    data[i], (shifts[i] * self.bins_per_semitone, 0))

        return new_data, targets


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
