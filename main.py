import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss

###
# w is w.T

# make layers
layers = []
layers.append(Layer(input_dimension=2, output_dimension=25, activation='ReLU'))
layers.append(Layer(input_dimension=25, output_dimension=50, activation='ReLU'))
layers.append(Layer(input_dimension=50, output_dimension=50, activation='ReLU'))
layers.append(Layer(input_dimension=50, output_dimension=25, activation='ReLU'))
layers.append(Layer(input_dimension=25, output_dimension=1, activation='sigmoid'))

def initialize(layers):
    params_values = {}
    np.random.seed(2)

    for index,layer in enumerate(layers):
        layer_idx = index + 1
        layer_input_size = layer.input_dimension
        layer_output_size = layer.output_dimension

        # layer.w = np.random.randn(
        #     layer_output_size, layer_input_size) * 0.1

        # layer.b = np.random.randn(
        #     layer_output_size, 1) * 0.1
        
        ## to be removed
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1

    return params_values

def forward_network(x_in_network, params_values,layers):
    memory = {}
    x_out = x_in_network

    for index,layer in enumerate(layers):
        layer_idx = index + 1
        x_in = x_out

        # to be removed
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        x_out, z = layer.forward(x_in, W_curr, b_curr)
        
        memory["A" + str(index)] = x_in
        memory["Z" + str(layer_idx)] = z

    return x_out, memory

def backward_network(Y_hat, Y, memory, params_values, layers, loss):
    grads_values = {}
    
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    
    dx_in = loss.backward()
    
    for layer_idx_prev, layer in reversed(list(enumerate(layers))):
        layer_idx_curr = layer_idx_prev + 1
        
        dx_out = dx_in
        
        x_in = memory["A" + str(layer_idx_prev)]
        z = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dx_in, dw, db = layer.backward(dx_out, W_curr, b_curr,z, x_in)
        
        grads_values["dW" + str(layer_idx_curr)] = dw
        grads_values["db" + str(layer_idx_curr)] = db
    return grads_values

def update(params_values ,grads_values, layers, learning_rate):
    for layer_idx, layer in enumerate(layers, 1):
        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)] 
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)] 
    return params_values

def train(X, Y, layers, epochs, learning_rate):

    params_values = initialize(layers)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cache = forward_network(X, params_values,layers)

        #### mine
        # softmax = Softmax()
        # Y_hat =  softmax.forward(Y_hat)

        loss = Loss(Y_hat, Y)
        cost = loss.forward()
        ########
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = backward_network(Y_hat, Y, cache, params_values,layers, loss)
        params_values = update(params_values ,grads_values, layers, learning_rate)

    return  params_values


####################
def load_data():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    train_data = unpickle('cifar-10-batches-py/data_batch_1')[b'data']
    train_labels = unpickle('cifar-10-batches-py/data_batch_1')[b'labels']
    test_data = unpickle('cifar-10-batches-py/test_batch')[b'data']
    test_labels = unpickle('cifar-10-batches-py/test_batch')[b'labels']
    return train_data, train_labels, test_data, test_labels


####### main
train_data, train_labels ,test_data ,test_labels  = load_data()

###########################################################
## to be changed

# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split

# def get_accuracy_value(Y_hat, Y):
#     Y_hat_ = convert_prob_into_class(Y_hat)
#     return (Y_hat_ == Y).all(axis=0).mean()

# def convert_prob_into_class(probs):
#     probs_ = np.copy(probs)
#     probs_[probs_ > 0.5] = 1
#     probs_[probs_ <= 0.5] = 0
#     return probs_

# # number of samples in the data set
# N_SAMPLES = 1000
# # ratio between training and test sets
# TEST_SIZE = 0.1

# X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

# # Training
# params_values =train(np.transpose(X_train), np.transpose(y_train.reshape((y_train.shape[0], 1))), layers, epochs=10000,learning_rate=0.01)

# # Prediction
# Y_test_hat, _ = forward_network(np.transpose(X_test), params_values,layers)

# # Accuracy achieved on the test set
# acc_test = get_accuracy_value(Y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
# print("Test set accuracy: {:.2f} - David".format(acc_test))