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

def feedforward(x):
    cache = []
    x_in = x
    cache.append(x_in)

    for index,layer in enumerate(layers):
        x_out = layer.forward(x_in)
        x_in = x_out
        cache.append(x_out)

    return cache

def backprop(y, cache):
    derivative_w = []    
    derivative_b = []

    loss = Loss(cache[-1], y)
    loss_value = loss.forward()

    print(loss_value)

    dA = loss.backward()
    for index,layer in reversed(list(enumerate(layers))[1:]):
        derivative_w.append(np.dot(cache[index].T,dA))
        derivative_b.append(np.sum(dA, axis=0, keepdims=True))

        dZ = np.dot(dA, layer.w.T) 
        dA = dZ * layer.backward(cache[index])

    derivative_w.append(np.dot(cache[0].T, dA))
    derivative_b.append(np.sum(dA, axis=0))

    derivative_w = derivative_w[::-1]
    derivative_b = derivative_b[::-1]

    return derivative_w, derivative_b

def update(lr, derivative_w, derivative_b):
    for index, layer in enumerate(layers):
        layer.w -= lr * derivative_w[index]
        layer.b -= lr * derivative_b[index]

def predict(data):
    cache = feedforward(data)
    return cache[-1].argmax()

def load_new_data():
    train_data, train_labels ,test_data ,test_labels  = load_data()
    train_data = make_feature_vector(train_data)
    test_data = make_feature_vector(test_data)
    train_labels = one_hot(np.asarray(train_labels),10)
    test_labels = one_hot(np.asarray(test_labels),10)

    np.savetxt('train_data.txt', train_data, fmt='%f')
    np.savetxt('test_data.txt', test_data, fmt='%f')
    np.savetxt('train_labels.txt', train_labels, fmt='%f')
    np.savetxt('test_labels.txt', test_labels, fmt='%f')

    return train_data, test_data, train_labels, test_labels

def load_data_from_file():
    train_data = np.loadtxt('train_data.txt', dtype=float)
    test_data = np.loadtxt('test_data.txt', dtype=float)
    train_labels = np.loadtxt('train_labels.txt', dtype=float)
    test_labels = np.loadtxt('test_labels.txt', dtype=float)

    return train_data, test_data, train_labels, test_labels



# train_data, test_data, train_labels, test_labels = load_new_data()

train_data, test_data, train_labels, test_labels = load_data_from_file()

x = train_data
y = np.array(train_labels)
lr = 0.5


# make layers
layers = []
layers.append(Layer(input_dimension=1024, output_dimension=128, activation='sigmoid'))
layers.append(Layer(input_dimension=128, output_dimension=128, activation='sigmoid'))
##
layers.append(Layer(input_dimension=128, output_dimension=10, activation='sigmoid'))

cache = []

epochs = 1500
for i in range(epochs):
    cache = feedforward(x)
    derivative_w, derivative_b = backprop(y, cache)
    update(lr, derivative_w, derivative_b)

def get_acc(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = predict(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100
	
# print("Training accuracy : ", get_acc(x_train, np.array(y_train)))
# print("Test accuracy : ", get_acc(x_val, np.array(y_val)))
