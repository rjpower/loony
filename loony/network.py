import numpy as np

class Network(object):
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def fprop(self, in_batch):
        for layer in self.layers:
            in_batch = layer.fprop(in_batch)

    def bprop(self, d_cost_d_out):
        for layer in reversed(self.layers):
            d_cost_d_out = layer.bprop(d_cost_d_out)

    def update(self):
        for layer in self.layers:
            layer.update()

    def train(self, input_reader, n_batches=10**4):
        correct_count = 0

        for i in range(n_batches):
            data, correct = input_reader.next()

            # predictions is a #classes vector
            cost_layer = self.layers[-1]
            cost_layer.set_actual(correct)

            self.fprop(data)
            predictions = cost_layer.predictions
            errors = cost_layer.errors
            if np.argmax(predictions) == correct:
                correct_count += 1

            if i % 100 == 0:
                print 'batch.%05d: P(correct)=%s' % (i, correct_count / 100.)
                correct_count = 0
            # print('Last example:', predictions, correct, errors)
            # d_cost_d_p = -2 * errors
            self.bprop(errors)
            self.update()