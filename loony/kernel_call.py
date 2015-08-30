import pyopencl as cl

ctx = cl.create_some_context()
default_queue = cl.CommandQueue(ctx)

class KernelCall(object):
    def __init__(self, kernel, queue=default_queue):
        self._queue = queue
        # kernel = lp.set_options(kernel, 'write_cl')
        self._kernel = kernel

    def __call__(self, *args, **kw):
        evt, result = self._kernel(self._queue, *args, **kw)
        if len(result) == 1:
            return result[0]
        return result
