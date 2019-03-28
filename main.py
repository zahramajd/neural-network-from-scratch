import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss

###
# w is w.T
# loss & softmax aren't layer

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

def backward_network(Y_hat, Y, memory, layers):
    grads_values = {}
    
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    
    dx_in = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat));
    
    for layer_idx_prev, layer in reversed(list(enumerate(layers))):
        layer_idx_curr = layer_idx_prev + 1
        
        dx_out = dx_in
        
        x_in = memory["A" + str(layer_idx_prev)]
        z = memory["Z" + str(layer_idx_curr)]

        dx_in, dw, db = layer.backward(dx_out, z, x_in)
        
        grads_values["dW" + str(layer_idx_curr)] = dw
        grads_values["db" + str(layer_idx_curr)] = db
    return grads_values

def train(X, layers):
    params_values = initialize(layers)

    return

## to be changed

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()