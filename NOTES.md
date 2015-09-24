# Loony Notes

## Theano Integration

After looking over the Theano code and documentation briefly, it seems like adding a
new backend should be fairly straightforward.  After loopy has performed it’s
graph optimizations, it dispatches the graph to what it calls a "linker"; the
linker is responsible for returning a set of thunk objects which can be executed
to actually perform operations.

I need to learn more about the actual Theano graph language to figure out the
complexities of the code generation.  A sample from their documentation:

```
  x, y = Variable(Double), Variable(Double)
  e = x + y
  fgraph = FunctionGraph([x, y], [e])
  fn, (new_x, new_y), (new_e, ) = MyLinker(fgraph).make_thunk(inplace)
  new_x.data = 1.0
  new_y.data = 2.0
  fn()
  print new_e.data # 3.0
  print e.data # 3.0 iff inplace == True (else unknown)
```

What's not totally clear is where/if Theano puts function boundaries: does it
generate one giant graph node before handing it to the linker, or is the linker
responsible for combining together operations on it's own?

Once we understand the graph language, actually generating kernels should be
fairly straightforward.

Most Theano programs appear to perform batching explicitly, so we wouldn't need
to auto-batch, but we _should_ have the kernels optimize their behavior by
assuming the batch size parameter will always be the same.

## Optimization

I made a quick pass at writing an optimizer; it does a simple brute-force pass
over possible unrolling options.  We can also add pre-fetching as well to this.

Still missing is the application of the appropriate assumptions on array sizes
to remove extra checks.
