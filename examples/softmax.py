import numpy as np
import loopy as lp
import pyopencl as cl


def softmax(queue, input_array):
    '''
    Compute the softmax of `input_array` (e^x_i) / sum(e^x_i for all i)self.
    '''
    exp_kernel = lp.make_kernel(
      name="exp",
      domains=[
        '{ [i]: 0<=i<n }',
      ],
      instructions=[
        'out_exp[i] = z[i]',
      ],
      assumptions="n>0")

    sum_kernel = lp.make_kernel(
      name="sum",
      domains=[
        '{ [i]: 0<=i<n }',
      ],
      instructions=[
        'out_sum = sum(i, exp[i])',
      ],
      assumptions='n>0')

    softmax_divide_kernel = lp.make_kernel(
      name="softmax_divide",
      domains=[
        '{ [i]: 0<=i<n }',
      ],
      instructions=[
        'out_final[i] = exp[i] / total',
      ],
      assumptions='n>0')
    print(exp_kernel)
    print(sum_kernel)
    print(softmax_divide_kernel)
    evt, (exp,) = exp_kernel(queue, z=input_array, E=2.7)
    evt, (sum_exp,) = sum_kernel(queue, exp=exp)
    evt, (softmax,) = softmax_divide_kernel(queue, exp=exp, total=sum_exp)

    return softmax

if __name__ == "__main__":
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    x = np.array([1, 2, 3])
    y = softmax(queue, x)
    print("-- x: %s" % (x,))
    print("-- y: %s" % (y,))
