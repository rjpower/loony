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
    "out[i] = a[i] + 1e-12"
    )

knl = lp.set_options(knl, write_cl=True)
evt, (out,) = knl(queue, a=a)
