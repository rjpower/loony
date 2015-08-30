import math

import loopy as lp
import numpy as np

from .common import DEBUG
from .layer import Layer
from .kernel_call import KernelCall

def softmax_fprop():
    '''
    Compute the softmax of `input_array` (e^x_i) / sum(e^x_i for all i).
    '''
    knl = lp.make_kernel(
        """{ [i,j,k]:
            0<=i<n and
            0<=j<n and
            0<=k<n
         }
        """,
        instructions=[
            'exp[i] = E ** in_batch[i]',
            '<float32> total = sum(j, exp[j])',
            'out[k] = exp[k] / total'
        ],
        assumptions="n>0")
    return knl

def softmax_bprop():
    '''
    Softmax bprop:

    If you do this out, dOut/dIn is actually expressed most simply in the
    form of `out`:

    dOut_i/dIn_j = {
      In_j * (1 - In_j) [ i == j]
      - In_i * In_j [i != j]
    }

    As I don't know how to express a ternary condition like that in loopy,
    I'm factoring the i == j case out.

    In practice, we'd combine softmax with a log-likelihood cost function, which
    when multiplied together results in a very simple expression for dCost/dIn:
    (prediction_i - correct_i), e.g. the negative error for the prediction.
    '''
    return lp.make_kernel(
        """{ [i,j]: 0<=i<n and 0<=j<n } """,
        instructions=[
          'd_cost_d_in[i] = -d_cost_d_out[i] * out[i] + sum(j, -out[i] * out[j] * d_cost_d_out[j])'
        ],
        assumptions="n>0")


class Softmax(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(softmax_fprop())
        self._bprop_kernel = KernelCall(softmax_bprop())

    def _fprop(self, in_batch):
        exp, self.pred = self._fprop_kernel(in_batch=in_batch, E=np.float32(math.e))
        return self.pred

    def _bprop(self, d_out):
        d_cost_d_in = self._bprop_kernel(d_cost_d_out=d_out, out=self.pred)
        if DEBUG:
            print 'ERR', d_out
            print 'OUT', self.pred
            print 'COST/X', d_cost_d_in
        return d_cost_d_in

    def _update(self):
        pass
