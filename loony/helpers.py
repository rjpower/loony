from .network import Network
from .dense import Dense
from .softmax import Softmax
from .costs import MSE

def build_feedforward_network(
        hidden_sizes=[100],
        hidden_activation=None,
        output_size=10,
        output_activation=Softmax,
        cost=MSE):
    net = Network()
    for i, hidden_size in enumerate(hidden_sizes):
        net.add(Dense(hidden_size))
        if hidden_activation:
            net.add(hidden_activation())
    net.add(Dense(output_size))
    net.add(output_activation())
    net.add(cost())
    return net