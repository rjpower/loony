import numpy as np
import loopy as lp
import pyopencl as cl
import pyopencl.array

# Convolution over 4d batch (image, width, height, color)
# with a weight matrix of size (num_tiles, tile_w, tile_h)
def convolve(images, weights):
  knl = lp.make_kernel(
         '''
         { [x,y,tile,image,i,j,color]:
           0<=x<w and
           0<=y<h and
           0<=tile<num_tiles and
           0<=image<num_images and
           0<=i,j<tile_size and
           0<=color<num_colors}
         ''',
         '''
         out[image,tile,x,y] = input[image,x+i,y+j] * weights[tile,i,j]
         ''',
         assumptions="w,h,num_tiles,tile_size,num_images,num_tiles>0 and w mod 16 = 0 and h mod 16 = 0")


  knl = lp.set_loop_priority(knl, "image,tile,x,y,i,j")

  knl = lp.split_iname(knl,  "x", 4, outer_tag="g.0", inner_tag="l.0")
  knl = lp.split_iname(knl,  "y", 4, outer_tag="g.1", inner_tag="l.1")
  knl = lp.split_iname(knl,  "tile", 4, outer_tag="g.2", inner_tag="l.2")
  knl = lp.split_iname(knl,  "i", 4, inner_tag="unr")
  knl = lp.split_iname(knl,  "j", 4, inner_tag="unr")

  knl = lp.add_dtypes(knl, { "out": np.float32, "weights": np.float32, "input": np.float32 })
  knl = lp.set_options(knl, 'write_cl')

  evt, (out,) = knl(queue, input=images, weights=weights)
  return out
