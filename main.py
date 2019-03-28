import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss

###
# w is w.T

# make layers
layers = []
layers.append(Layer(input_dimension=2, output_dimension=4, activation='ReLU'))
layers.append(Layer(input_dimension=4, output_dimension=6, activation='ReLU'))
layers.append(Layer(input_dimension=6, output_dimension=6, activation='ReLU'))
layers.append(Layer(input_dimension=6, output_dimension=4, activation='ReLU'))
layers.append(Layer(input_dimension=4, output_dimension=1, activation='sigmoid'))


def initialize(layers):
    params_values = {}
    np.random.seed(2)

    for index,layer in enumerate(layers):
        layer_idx = index + 1
        layer_input_size = layer.input_dimension
        layer_output_size = layer.output_dimension

        layer.w = np.random.randn(
            layer_output_size, layer_input_size) * 0.1

        layer.b = np.random.randn(
            layer_output_size, 1) * 0.1
        
        ## to be removed
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values

def train():
    params_values = initialize(layers)

    return

train()