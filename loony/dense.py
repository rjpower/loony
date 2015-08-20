import loopy as lp
import numpy as np

from .layer import Layer
from .kernel_call import KernelCall

def dense_fprop():
    """
    Forward propagation:

    for each output[i]
        out[i] = sum(j, w[i, j] * in[j]) + bias[i]
    """
    return lp.make_kernel(
        domains="""
         { [i, j]:
           0<=i<input_len and
           0<=j<output_len
         }
         """,
        instructions="""
            out[j] = sum(i, weights[i, j] * in_batch[i]) + bias[j]
         """,
        assumptions="output_len,input_len>0"
    )

def dense_bprop_weights():
    """
    Given: dCost/dOut
    Compute: dCost/dW using the chain rule.

    N.B. dOut_x/dW_yz is zero for d(dCost_x)/dWyz where x != z

    Sum over instance gradients.
    """
    return lp.make_kernel(
        domains="""
           { [i, j]:
             0<=i<input_len and
             0<=j<output_len
           }
         """,
        instructions="""
           d_cost_d_w[i, j] = d_cost_d_out[j] * in_batch[i]
        """,
        assumptions="output_len,input_len>0"
    )

def dense_bprop_bias():
    """
    Given: dCost/dOut
    Compute: dCost/dBias.

    Sum over instance gradients.
    """
    return lp.make_kernel(
        domains=""" { [j]: 0<=j<output_len } """,
        instructions=[
            """
            d_cost_d_bias[j] = d_cost_d_out[j]
            """
        ],
        assumptions="output_len>0"
    )

def dense_bprop_input():
    """
    Given: dCost/dOut
    Compute: dCost/dIn via dCost/dOut * dOut/dIn.

    dOut/dIn is just the weight matrix.
    dOut/dIn[i, j] = w[i,j]
    """
    knl = lp.make_kernel(
        domains="""
        { [i, j]:
           0<=i<input_len and
           0<=j<output_len
        }
        """,
        instructions=[
            """
            d_cost_d_in[i] = sum(j, d_cost_d_out[j] * weights[i, j])
            """,
        ],
        assumptions="output_len,input_len>0"
    )
    knl = lp.set_options(knl, 'write_cl')
    return knl

class Dense(Layer):
    def __init__(self, output_size, learning_rate=0.01):
        self._output_size = output_size
        self.learning_rate = learning_rate
        self._weights = None
        self._bias = None

        self._fprop_kernel = KernelCall(dense_fprop())
        self._bprop_w = KernelCall(dense_bprop_weights())
        self._bprop_b = KernelCall(dense_bprop_bias())
        self._bprop_in = KernelCall(dense_bprop_input())

    def _fprop(self, in_batch):
        if self._weights is None:
            # initialize weights and biases from a normal distribution
            # scale down to keep avg(output) = avg(input)
            self._weights = np.random.randn(
                in_batch.shape[0],
                self._output_size).astype(np.float32)
            self._weights /= (self._output_size * in_batch.shape[0])
            self._bias = np.random.randn(self._output_size).astype(np.float32) / self._output_size

        self._in_batch = in_batch
        return self._fprop_kernel(weights=self._weights, in_batch=in_batch, bias=self._bias)

    def _bprop(self, d_out):
        self._w_grad = self._bprop_w(d_cost_d_out=d_out, in_batch=self._in_batch)
        self._b_grad = self._bprop_b(d_cost_d_out=d_out)
        result = self._bprop_in(d_cost_d_out=d_out, weights=self._weights)
        return result

    def _update(self):
        self._weights -= self._w_grad * self.learning_rate
        self._bias -= self._b_grad * self.learning_rate
