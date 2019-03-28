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
        # params_values['W' + str(layer_idx)] = np.random.randn(
        #     layer_output_size, layer_input_size) * 0.1
        # params_values['b' + str(layer_idx)] = np.random.randn(
        #     layer_output_size, 1) * 0.1

    return 

def forward_network(x_in_network,layers):
    memory = {}
    x_out = x_in_network

    for index,layer in enumerate(layers):
        layer_idx = index + 1
        x_in = x_out
        
        # to be removed
        # activ_function_curr = layer.activation
        # W_curr = params_values["W" + str(layer_idx)]
        # b_curr = params_values["b" + str(layer_idx)]
        # A_curr, Z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)

        x_out, z = layer.forward(x_in)
        
        memory["A" + str(index)] = x_in
        memory["Z" + str(layer_idx)] = z

    return x_out, memory

def train(X, layers):
    params_values = initialize(layers)

    return



from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_SAMPLES = 1000
TEST_SIZE = 0.1
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

train(X_train, layers)