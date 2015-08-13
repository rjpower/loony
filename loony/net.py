#!/usr/bin/env python

import loopy as lp
import numpy as np
import pyopencl as cl
import numpy as np
import math

import pymbolic.primitives as P

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

EPSILON = 0.01

class KernelCall(object):
    def __init__(self, queue, kernel):
        self._queue = queue
        #kernel = lp.set_options(kernel, 'write_cl')
        self._kernel = kernel

    def __call__(self, *args, **kw):
        evt, result = self._kernel(self._queue, *args, **kw)
        if len(result) == 1:
            return result[0]
        return result

def fprop_cost():
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

def fc_fprop():
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

def fc_bprop_weights():
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

def fc_bprop_bias():
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

def fc_bprop_input():
    """
    Given: dCost/dOut
    Compute: dCost/dIn via dCost/dOut * dOut/dIn.
    dOut/dIn[i, j] = w[i,j]

    dOut/dIn is just the weight matrix!
    """
    return lp.make_kernel(
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

    knl = lp.set_options(knl, 'write_cl')
    return knl

def softmax_bprop():
    return lp.make_kernel(
        """{ [i]: 0<=i<n } """,
        instructions=[
            'out[i] = d_cost_d_out[i]'
        ],
        assumptions="n>0")

class Reader(object):
    def __init__(self, np_array):
        self._data = np_array
        self._idx = 0

    def next(self):
        idx = self._idx
        self._idx += 1
        data = self._data.data[idx].astype(np.float32)
        data = (data / 16.0) - 0.5
        classes = self._data.target[idx]
        return data, classes

class Network(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fprop(self, in_batch):
        for layer in self.layers:
            #print 'IN:', in_batch
            in_batch = layer.fprop(in_batch)

    def bprop(self, cost):
        for layer in reversed(self.layers):
            cost = layer.bprop(cost)

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, input_reader):
        for i in range(1000):
            data, correct = input_reader.next()

            # predictions is a #classes vector
            cost_layer = self.layers[-1]
            cost_layer.set_actual(correct)

            self.fprop(data)
            predictions = cost_layer.predictions
            errors = cost_layer.errors
            if i % 10 == 0:
                print 'P:', predictions
                print 'Correct?', np.argmax(predictions), correct, 'Total error:', np.sum(errors ** 2)
            self.bprop(errors)
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

        self._fprop_kernel = KernelCall(queue, fc_fprop())
        self._bprop_w = KernelCall(queue, fc_bprop_weights())
        self._bprop_b = KernelCall(queue, fc_bprop_bias())
        self._bprop_in = KernelCall(queue, fc_bprop_input())

    def _fprop(self, in_batch):
        if  self._weights is None:
            self._weights = np.random.randn(in_batch.shape[0], self._output_size).astype(np.float32) / (
                    self._output_size * in_batch.shape[0])
            self._bias = np.random.randn(self._output_size).astype(np.float32) / self._output_size

        self._in_batch = in_batch
        return self._fprop_kernel(weights=self._weights, in_batch=in_batch, bias=self._bias)

    def _bprop(self, d_out):
        self._w_grad = self._bprop_w(d_cost_d_out=d_out, in_batch=self._in_batch)
        self._b_grad = self._bprop_b(d_cost_d_out=d_out)
        return self._bprop_in(d_cost_d_out=d_out, weights=self._weights)

    def _update(self, epsilon=EPSILON):
        self._weights -= self._w_grad * epsilon
        self._bias -= self._b_grad * epsilon

class SoftMaxLayer(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(queue, softmax_fprop())
        self._bprop_kernel = KernelCall(queue, softmax_bprop())

    def _fprop(self, in_batch):
        exp, pred =  self._fprop_kernel(in_batch=in_batch, E=np.float32(math.e))
        return pred

    def _bprop(self, d_out):
        return self._bprop_kernel(d_cost_d_out=d_out)

    def _update(self):
        pass

class CostLayer(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(queue, fprop_cost())
        #self._bprop_kernel = KernelCall(queue, bprop_cost())

    def set_actual(self, actual):
        self.actual = actual

    def _fprop(self, in_batch):
        correct = np.zeros(in_batch.shape).astype(np.float32)
        correct[self.actual] = 1

        self.predictions = in_batch

        self.errors = self._fprop_kernel(in_batch=in_batch, correct=correct)
        return self.errors

    def _bprop(self, d_out):
        return d_out

    def _update(self):
        pass


def build(hidden1_size=100, hidden2_size=10):
    net = Network()
    net.add(FullyConnectedLayer(hidden1_size))
    net.add(FullyConnectedLayer(hidden2_size))
    net.add(SoftMaxLayer())
    net.add(CostLayer())
    return net

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    digits = load_digits()
    network = build()
    network.train(Reader(digits))
