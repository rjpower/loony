from __future__ import print_function

import time
import numpy as np
import loopy as lp
import pyopencl as cl
from loony.util import time_op

class Optimizer(object):
    def __init__(self, knl, queue, size_hints):
        self._knl = knl
        self._queue = queue
        self._size_hints = size_hints
        self._example_data = self._make_data(size_hints)

    def _make_data(self, size_hints):
        # we could try to guess size hints for inputs if
        # unspecified...
        #inputs = [
        #    name for name in self._knl.get_read_variables()
        #    if name in self._knl.arg_dict
        #]

        data = {}
        for name, shape in size_hints.iteritems():
            data[name] = cl.array.zeros(self._queue, shape, dtype=np.float32)

        return data

    def _test(self, knl):
        #knl = lp.set_options(knl, 'write_cl')
        # Run the operation twice; the first time we actually generate the
        # kernel and compiled it, which we don't want to charge for.
        # TODO: generate kernel explicitly, instead of running the operation twice...
        knl(self._queue, **self._example_data)
        st = time.time()
        evt, result = knl(self._queue, **self._example_data)
        evt.wait()
        ed = time.time()
        return ed - st

    def opt(self, knl, inames):
        if not inames:
            elapsed = self._test(knl)
            return (knl, [], elapsed)

        iname = inames[0]
        inames = inames[1:]
        best = (None, [], 1e6)
        for unroll_factor in [2, 4, 8]:
            split_knl = lp.split_iname(knl, iname, unroll_factor, inner_tag="unr")
            # print('DIM', knl.cache_manager.dim_min(0))
            # TODO: find mapping from iname to the domain variable that
            # constraint it.  I'd like to inject assumptions on the size of the
            # input, e.g.
            # split_knl = lp.assume(split_knl, 'input_len mod %d = 0' % unroll_factor)

            # we track the choices made by the optimizer in a silly way for now
            res, choices, elapsed = self.opt(split_knl, inames)
            print('Split:', iname, elapsed)
            if elapsed < best[2]:
                best = (res,
                        choices + ['iname=%s unroll=%d ' % (iname, unroll_factor)],
                        elapsed)

        return best

def optimize(knl, queue, size_hints={}):
    optimizer = Optimizer(knl, queue, size_hints)
    knl, choices, elapsed = optimizer.opt(knl, list(knl.all_inames()))
    return knl, choices

