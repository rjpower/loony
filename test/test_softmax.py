import loopy as lp
import numpy as np
import pyopencl as cl
import math

knl = lp.make_kernel(
         """{ [i,j,k]:
             0<=i<n and
             0<=j<n and
             0<=k<n
          }
         """,
         instructions=[
             '<float32>exp[i] = E ** in_batch[i]',
             '<float32> total = sum(j, exp[j])',
             'out[k] = exp[k] / total'
         ],
         assumptions="n>0")

knl = lp.set_options(knl, 'write_cl')

if __name__ == "__main__":
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    ex = math.e ** x
    print 'np softmax:', x / ex.sum()
    print knl
    evt, out = knl(queue, in_batch=x, E=np.float32(math.e))
    print 'lp sofxmax:', out
