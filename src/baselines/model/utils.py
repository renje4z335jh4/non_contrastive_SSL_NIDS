from typing import List
from torch import nn

activation_mapper = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid()
}

def create_network(layers: List[List]):
    """
    This function builds a linear model whose units and layers depend on
    the passed @layers argument
    :param layers: a list of tuples indicating the layers architecture (in_neuron, out_neuron, activation_function)
    :return: a fully connected neural net (Sequentiel object)
    """
    net_layers = []
    for in_neuron, out_neuron, act_fn in layers:
        net_layers.append(nn.Linear(in_neuron, out_neuron))
        if act_fn:
            net_layers.append(act_fn)
    return nn.Sequential(*net_layers)
