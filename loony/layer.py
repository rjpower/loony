class Layer(object):
    def fprop(self, in_batch):
        return self._fprop(in_batch)

    def bprop(self, d_out):
        # print d_out
        return self._bprop(d_out)

    def update(self):
        return self._update()

    def __str__(self):
        return "%s()" % (self.__class__.__name__,)

    def __repr__(self):
        return str(self)

