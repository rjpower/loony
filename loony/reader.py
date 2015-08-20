import numpy as np

class Reader(object):
    def __init__(self, bunch):
        np.set_printoptions(precision=4, linewidth=120)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(bunch.data))
        self._data = bunch.data[shuffle_idx].astype(np.float32)
        self._target = bunch.target[shuffle_idx]

        # de-mean data and normalize to [-1, 1]
        self._data -= np.mean(self._data)
        self._data /= np.max(np.abs(self._data))

        self._idx = 0

    def next(self):
        data = self._data[self._idx]
        classes = self._target[self._idx]
        self._idx += 1
        if self._idx >= len(self._data):
            self._idx = 0
        return data, classes
