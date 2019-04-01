import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss


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

def make_feature_vector(data):
    feature_vector = np.zeros((10000, 1024))
    row_index = 0
    for row in  data:
        for i in range(0, 1024):
            avg = (row[i] + row[i+1024] + row[i+2048])/3
            feature_vector[row_index][i] = avg
        row_index += 1
    return feature_vector

def one_hot(a, num_classes):
      return np.squeeze(np.eye(num_classes)[a.reshape(-1)])



# train_data, train_labels ,test_data ,test_labels  = load_data()

# train_data = make_feature_vector(train_data)
# test_data = make_feature_vector(test_data)

# train_labels = one_hot(np.asarray(train_labels),10)
# test_labels = one_hot(np.asarray(test_labels),10)


import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def feedforward(x):
    cache = []
    x_in = x
    for index,layer in enumerate(layers):
        x_out = layer.forward(x_in)
        x_in = x_out
        cache.append(x_out)

    return cache

def backprop(y, cache):
    loss = Loss(cache[-1], y)
    loss_value = loss.forward()

    print('Error :', loss_value)

    return

dig = load_digits()
onehot_target = pd.get_dummies(dig.target)
x_train, x_val, y_train, y_val = train_test_split(dig.data, onehot_target, test_size=0.1, random_state=20)

x = x_train/16.0
y = np.array(y_train)
lr = 0.5


# make layers
layers = []
layers.append(Layer(input_dimension=64, output_dimension=128, activation='sigmoid'))
layers.append(Layer(input_dimension=128, output_dimension=128, activation='sigmoid'))
##
layers.append(Layer(input_dimension=128, output_dimension=10, activation='sigmoid'))


cache = []

epochs = 150
for i in range(epochs):
    cache = feedforward(x)
    backprop(y, cache)


