# Example of something with convolutions in loopy:
# https://github.com/inducer/loopy/blob/master/test/test_loopy.py#L1205

# This paper could be helpful in trying to understand more complicated
# dependency structures in loopy:
# http://arxiv.org/abs/1405.7470

import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array  # noqa

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 200
a = cl.array.arange(queue, n, dtype=np.float32)

knl = lp.make_kernel(
    "{ [i,k]: 0<=i,k<n }",
    "out[i] = sum(k, tanh(a[k])*a[i])"
    # "out[i] = 2*a[i]"
    )

knl = lp.set_options(knl, write_cl=True)
evt, (out,) = knl(queue, a=a)

from diff_kernel import diff_kernel

diffd_knl = diff_kernel(knl, ("output"), "a")

print diffd_knl

evt, (out_diff,) = diffd_knl(queue, a=a)
