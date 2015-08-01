import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array

def softmax(input_array):
  '''
  Compute the softmax of `input_array` (e^x_i) / sum(e^x_i for all i).
  '''
  exp_kernel = lp.make_kernel(
      ''' { [i]: 0<=i<n } ''',
     ''' out[i] = pow(E, z[i])''',
     assumptions="n>0")

  sum_kernel = lp.make_kernel(
      '{ [i]: 0<=i<n }',
      'out = sum(i, exp[i])',
      assumptions='n>0]')

  softmax_kernel = lp.make_kernel(
      '{ [i]: 0<=i<n }',
      'out[i] = exp[i] / total',
      assumptions='n>0')

  evt, (exp,) = exp_kernel(z=input_array)
  evt, (sum_exp,) = sum_kernel(exp=exp)
  evt, (softmax,) = softmax_kernel(exp=exp, total=sum_exp)

  return softmax

def tanh_activation(input_array):
  return _activation('tanh', input_array)

def relu_activation(input_array):
  return _activation('relu', input_array)
