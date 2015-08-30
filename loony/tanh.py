import loopy as lp

from .common import DEBUG
from .layer import Layer
from .kernel_call import KernelCall

def tanh_fprop():
    '''
    Compute the elementwise tanh of input vector
    '''
    knl = lp.make_kernel(
        """{ [i]:
            0<=i<n
         }
        """,
        instructions=[
            'out[i] = tanh(in_batch[i])'
        ],
        assumptions="n>0")
    return knl

def tanh_bprop():
    '''
    Tanh bprop:

        tanh(u)' = 1 - tanh(u)**2

    Chained together with the gradient dCost/du, we get ... ?

    TODO: figure out if just multiplying with dCost/dOut_i is the right thing
    here.

    '''
    return lp.make_kernel(
        """{ [i]: 0<=i<n } """,
        instructions=[
          'd_cost_d_in[i] = d_cost_d_out[i] * (1.0 - out[i] ** 2)'
        ],
        assumptions="n>0")


class Tanh(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(tanh_fprop())
        self._bprop_kernel = KernelCall(tanh_bprop())

    def _fprop(self, in_batch):
        self.pred = self._fprop_kernel(in_batch=in_batch)
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
