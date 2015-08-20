import numpy as np
import loopy as lp

from .layer import Layer
from .kernel_call import KernelCall

def fprop_mse_cost():
    """
    Compute the cross-entropy cost for a given set of predictions.

    Predictions: (#classes) -- probability vector per instance
    Actual: (#classes) -- 1 if the correct class, else 0

    Error = sum(i, i == actual ? p_i : 1 - p_i)
    """
    return lp.make_kernel(
        '{ [i]: 0<=i<classes }',
        [
            'cost[i] = correct[i] - in_batch[i]'
        ],
        assumptions="classes>=0"
    )

class MSE(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(fprop_mse_cost())

    def set_actual(self, actual):
        self.actual = actual

    def _fprop(self, in_batch):
        correct = np.zeros(in_batch.shape).astype(np.float32)
        correct[self.actual] = 1

        self.predictions = in_batch
        self.errors = self.predictions - correct
        self.cost = np.sum(self.errors ** 2)
        return self.errors

    def _bprop(self, errors):
        return -2 * self.errors

    def _update(self):
        pass
