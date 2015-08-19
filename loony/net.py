#!/usr/bin/env python

import loopy as lp
import numpy as np
import pyopencl as cl
import numpy as np
import math

import pymbolic.primitives as P

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

NUM_BATCHES = 10000
DEBUG = False
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
         # 'd_cost_d_in[i] = -d_cost_d_out[i]'
        ],
        assumptions="n>0")

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

class Network(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fprop(self, in_batch):
        for layer in self.layers:
            #print 'IN:', in_batch
            in_batch = layer.fprop(in_batch)

    def bprop(self, d_cost_d_out):
        for layer in reversed(self.layers):
            d_cost_d_out = layer.bprop(d_cost_d_out)

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, input_reader):
        correct_count = 0

        for i in range(NUM_BATCHES):
            data, correct = input_reader.next()

            # predictions is a #classes vector
            cost_layer = self.layers[-1]
            cost_layer.set_actual(correct)

            self.fprop(data)
            predictions = cost_layer.predictions
            errors = cost_layer.errors
            if np.argmax(predictions) == correct:
                correct_count += 1

            if i % 100 == 0:
                print 'batch.%05d: P(correct)=%s' % (i, correct_count/100.)
                correct_count = 0
            #print 'Last example:', predictions, correct, errors

            d_cost_d_p = -2 * errors
            self.bprop(errors)
            self.update()



class Layer(object):
    def fprop(self, in_batch):
        return self._fprop(in_batch)

    def bprop(self, d_out):
        #print d_out
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
            # initialize weights and biases from a normal distribution
            # scale down to keep avg(output) = avg(input)
            self._weights = np.random.randn(in_batch.shape[0], self._output_size).astype(np.float32) / (
                    self._output_size * in_batch.shape[0])
            self._bias = np.random.randn(self._output_size).astype(np.float32) / self._output_size

        self._in_batch = in_batch
        return self._fprop_kernel(weights=self._weights, in_batch=in_batch, bias=self._bias)

    def _bprop(self, d_out):
        self._w_grad = self._bprop_w(d_cost_d_out=d_out, in_batch=self._in_batch)
        self._b_grad = self._bprop_b(d_cost_d_out=d_out)
        result = self._bprop_in(d_cost_d_out=d_out, weights=self._weights)
        return result

    def _update(self, epsilon=EPSILON):
        self._weights -= self._w_grad * epsilon
        self._bias -= self._b_grad * epsilon

class SoftMaxLayer(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(queue, softmax_fprop())
        self._bprop_kernel = KernelCall(queue, softmax_bprop())

    def _fprop(self, in_batch):
        exp, self.pred =  self._fprop_kernel(in_batch=in_batch, E=np.float32(math.e))
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

class CostLayer(Layer):
    def __init__(self):
        self._fprop_kernel = KernelCall(queue, fprop_cost())

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


def build(hidden1_size=250, hidden2_size=10):
    net = Network()
    net.add(FullyConnectedLayer(hidden1_size))
    if hidden2_size > 0:
        net.add(FullyConnectedLayer(hidden2_size))
    net.add(SoftMaxLayer())
    net.add(CostLayer())
    return net

if __name__ == "__main__":
    from sklearn.datasets import load_digits, load_iris
    digits = load_digits()
    iris = load_iris()
    network = build(hidden2_size=10)
    network.train(Reader(digits))
