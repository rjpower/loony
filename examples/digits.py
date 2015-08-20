#!/usr/bin/env python

from sklearn.datasets import load_digits
from loony import build_feedforward_network, Reader, Softmax, Tanh

if __name__ == "__main__":
    digits = load_digits()
    network = build_feedforward_network(
        hidden_sizes=[100],
        hidden_activation=Tanh,
        output_activation=Softmax,
        output_size=10)
    network.train(Reader(digits))

