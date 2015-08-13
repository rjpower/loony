#!/usr/bin/env python

import loopy as lp
import numpy as np


EPSILON = 0.001

def make_cost_kernel():
    """
    Predictions: (#instances, #classes) -- probability vector per instance
    Actual: (#instances) -- index of the correct answer

    For an instance:

    Error = sum(i, i == actual ? p_i : 1 - p_i)
    """
    return lp.make_kernel(
        domains="""
          { [instance, i]:
             0<=instance<batch_size and
             0<=j<classes and
          }
        """,
        instructions="""
           cost[instance] = sum(j,
             j == actual[instance] ?
               1 - prediction[instance, j] :
               prediction[instance, j]);
        """,
        assumptions="instance,j>=0"
    )

def fc_fprop():
    """
    Forward propagation:

    for each output[instance, i]
        out[instance, i] = sum(j, w[i, j] * in[instance, j]) + bias[i]
    """
    return lp.make_kernel(
        domains="""
         { [instance, i, j]:
           0<=instance<batch_size and
           0<=i<output_len and
           0<=j<input_len
         }
         """,
        instructions="""
            out[instance, i] = sum(j, w_matrix[i, j] * in_v[instance, j]) + bias[i]
         """,
        assumptions="instance,i,j>=0"
    )

def fc_bprop_weights():
    """
    Given: dCost/dOut
    Compute: dCost/dW using the chain rule.

    N.B. dOut_x/dW_yz is zero for d(dCost_x)/dWyz where x != y

    Sum over instance gradients.
    """
    return lp.make_kernel(
        domains="""
           { [instance, i, j]:
             0<=instance<batch_size and
             0<=i<output_len and
             0<=j<input_len
           }
         """,
        instructions="""
           d_out_d_w[i, j] = sum(instance, in_v[instance, j])
           d_cost_d_w[i, j] = d_cost_d_out[i] * d_out_d_w[i,j]
        """,
        assumptions="instance,i,j>=0"
    )

def fc_bprop_bias():
    """
    Given: dCost/dOut
    Compute: dCost/dBias.

    Sum over instance gradients.
    """
    return lp.make_kernel(
        domains="""
         { [instance, i, j]:
           0<=instance<batch_size and
           0<=i<output_len and
           0<=j<input_len
         }
        """,
        instructions=[
            """
            d_cost_d_bias[i] = sum(instance, d_cost_d_out[i])
            """
        ],
        assumptions="instance,i,j>=0"
    )

def fc_bprop_input(out_v, w_matrix):
    """
    Given: dCost/dOut
    Compute: dCost/dIn via dCost/dOut * dOut/dIn.
    dOut/dIn[i, j] = w[i,j]
    """
    return lp.make_kernel(
        domains="""
        { [instance, i, j]:
           0<=instance<batch_size and
           0<=i<input_len and
           0<=j<output_len
        }
        """,
        instructions=[
            """
            d_cost_d_in[instance, i] = sum(j, d_cost_d_out[instance, j] * w_matrix[i, j])
            """,
        ],
        assumptions="instance,i,j>=0"
    )

def softmax_fprop():
    '''
    Compute the softmax of `input_array` (e^x_i) / sum(e^x_i for all i).
    '''
    return lp.make_kernel(
        """{ [instance, i]:
            0<=i<n  and
            0<=instance<batch_size
         }
        """,
        instructions=[
            'exp[instance, i] = E ** input_v[instance, i]',
            'total[instance] = sum(i, exp)',
            'out[instance, i] = exp[instance, i] / total[instance]'
        ],
        assumptions="n>0")

def softmax_bprop():
    """TODO"""
    pass


class Network(object):
    def __init__(self):
        self.layers = []
        self.cost_kernel = make_cost_kernel()

    def add(self, layer):
        self.layers.append(layer)

    def fprop(self, in_batch):
        for layer in self.layers:
            in_batch = layer.fprop(in_batch)

    def bprop(self, cost):
        for layer in reversed(self.layers):
            cost = layer.bprop(cost)

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, input_reader):
        for i in range(1000):
            batch, correct = input_reader.next_batch()

            # predictions is a #instances*#classes matrix
            predictions = self.fprop(batch)

            error = self.cost_kernel(predictions, correct)

            self.bprop(error)
            self.update()


class Layer(object):
    def fprop(self, in_batch):
        return self._fprop(in_batch)

    def bprop(self, d_out):
        return self._bprop(d_out)

    def update(self):
        return self._update()


class FullyConnectedLayer(Layer):
    def __init__(self, output_size):
        self._output_size = output_size
        self._weights = None
        self._bias = None

    def _fprop(self, in_batch):
        return fc_fprop(in_batch, self._weights, self._bias)

    def _bprop(self, d_out):
        self._w_grad = fc_bprop_weights(d_out)
        self._b_grad = fc_bprop_bias(d_out)
        return fc_bprop_input(d_out)

    def _update(self, epsilon=EPSILON):
        self._weights -= self._w_grad * epsilon
        self._bias -= self._b_grad * epsilon

class SoftMaxLayer(Layer):
    def _fprop(self, in_batch):
        return softmax_fprop(in_batch)

    def _bprop(self, d_out):
        return softmax_bprop(d_out)

    def _update(self):
        pass


def build(input_size, hidden1_size=100, hidden2_size=10):
    # TODO: define InputLayer!
    net = InputLayer(input_size)
    net.add(FullyConnectedLayer(hidden1_size))
    net.add(FullyConnectedLayer(hidden2_size))
    net.add(SoftMaxLayer())
    return net

if __name__ == "__main__":
    network = build()
    x = np.array([1, 2, 3])
