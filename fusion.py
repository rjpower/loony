import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 15 * 10**6
a = cl.array.arange(queue, n, dtype=np.float32)

exp_kernel = lp.make_kernel(
     ''' { [i]: 0<=i<n } ''',
     ''' exp[i] = E ** z[i]''',
     assumptions="n>0")

sum_kernel = lp.make_kernel(
    '{ [j]: 0<=j<n }',
    'total = sum(j, exp[j])',
    assumptions='n>0')

softmax_kernel = lp.make_kernel(
    '{ [k]: 0<=k<n }',
    'out3[k] = exp[k] / total',
    [
        lp.GlobalArg("total", None, shape=()),
        "..."
        ],
    assumptions='n>0')

big_honkin_knl = lp.fuse_kernels([exp_kernel, sum_kernel, softmax_kernel])
#big_honkin_knl = lp.fuse_kernels([exp_kernel, sum_kernel])

print softmax_kernel.arg_dict["total"].shape

#big_honkin_knl = lp.tag_inames(big_honkin_knl, dict(i="l.0"))

big_honkin_knl = lp.set_options(big_honkin_knl, write_cl=True)



a = np.random.randn(20)
big_honkin_knl = big_honkin_knl(queue, z=a, E=np.float64(5))
