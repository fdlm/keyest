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
            new_data[i] = shift(
                data[i], (shifts[i] * self.bins_per_semitone, 0))

        return new_data, targets
