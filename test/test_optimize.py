#!/usr/bin/env python

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array
import islpy

from loony.optimize import optimize
from loony.dense import dense_fprop
from loony.util import time_op

N = 1024

def test_optimization():
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)

    knl = dense_fprop()

    # N.B. size_hints should eventually be either specified by the user or
    # inferred at run-time.  (For example, we could defer optimization until
    # we are called for the first time, and round-up the size of the "real"
    # input arguments to the nearest power of 2).
    optimized_knl, choices=\
            optimize(knl, queue=queue, size_hints={ 'in_batch': (N,), 'weights': (N, N), 'bias': N })

    in_batch = np.ones(N, dtype=np.float32)
    weights = np.ones((N, N), dtype=np.float32)
    bias = np.ones(N, dtype=np.float32)

    print 'Optimization choices:', choices
    for i in range(10):
        (evt, result), naive_time = time_op(
            lambda: knl(queue, in_batch=in_batch, weights=weights, bias=bias))
        evt.wait()
        (evt, result), optimized_time = time_op(
            lambda: optimized_knl(queue, in_batch=in_batch, weights=weights, bias=bias))
        evt.wait()
        print 'Time:', naive_time, optimized_time, 'Speedup:', naive_time / optimized_time

if __name__ == '__main__':
    test_optimization()
