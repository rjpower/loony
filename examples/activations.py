import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array

def _activation(operator, input_array):
  knl = lp.make_kernel(
         '''
#define relu(x) (max(0, x))
{ [i]: 0<=i<n }
''',
         ''' out[i] = %s(in[i]) ''' % operator,
         assumptions="n>0")

  knl = lp.split_iname(knl,  "i", 4, outer_tag="g.0", inner_tag="l.0")
  knl = lp.add_dtypes(knl, { "out": np.float32, "input_array": np.float32 })
  knl = lp.set_options(knl, 'write_cl')

def tanh_activation(input_array):
  return _activation('tanh', input_array)

def relu_activation(input_array):
  return _activation('relu', input_array)
