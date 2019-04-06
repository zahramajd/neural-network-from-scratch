import numpy as np
import matplotlib.pyplot as plt

from layer import Layer
from softmax import Softmax
from loss import Loss

def load_data():
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # train_data = unpickle('cifar-10-batches-py/data_batch_5')[b'data']
    # train_labels = unpickle('cifar-10-batches-py/data_batch_5')[b'labels']

    test_data = unpickle('cifar-10-batches-py/test_batch')[b'data']
    test_labels = unpickle('cifar-10-batches-py/test_batch')[b'labels']
    return test_data, test_labels

def load_new_data():
    test_data, test_labels  = load_data()

    # train_data = make_feature_vector(train_data)
    test_data = make_feature_vector(test_data)
    # train_labels = one_hot(np.asarray(train_labels),10)
    test_labels = one_hot(np.asarray(test_labels),10)

    # np.savetxt('converted_data/train_data_5.txt', train_data, fmt='%f')
    np.savetxt('converted_data/test_data.txt', test_data, fmt='%f')

    # np.savetxt('converted_data/train_labels_5.txt', train_labels, fmt='%f')
    np.savetxt('converted_data/test_labels.txt', test_labels, fmt='%f')

    return test_data, test_labels

def load_test_from_file():
    test_data = np.loadtxt('converted_data/test_data.txt', dtype=float)
    test_labels = np.loadtxt('converted_data/test_labels.txt', dtype=float)

    return  test_data, test_labels

def load_batch_data_from_file(index):
    train_data = np.loadtxt('converted_data/train_data_' + str(index) + '.txt', dtype=float)
    train_labels = np.loadtxt('converted_data/train_labels_' + str(index) + '.txt', dtype=float)

    return  train_data, train_labels

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

    # print(loss_value)

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

    return derivative_w, derivative_b, loss_value

def update(lr, derivative_w, derivative_b):
    for index, layer in enumerate(layers):
        layer.w -= lr * derivative_w[index]
        layer.b -= lr * derivative_b[index]

def plot_loss(losses, epochs):
    plt.plot(epochs, losses, color='red')
    
def Adam_optimizer(derivative, epoch, m, v, learning_rate = 0.01):
    
    beta1 = 0.9
    beta2 = 0.999

    opt_values = []
    for index, layer in enumerate(layers):
        m[index] = beta1 * m[index] + (1-beta1)* derivative[index]
        v[index] = beta2 * v[index] + (1-beta2) * (np.power(derivative[index], 2))

        mb = m[index] / (1- beta1**epoch)
        vb = v[index] / (1- beta2**epoch)

        opt_values.append(-learning_rate * mb / (np.sqrt(vb) + 1e-7))

    return opt_values, m, v

def Adagrad_optimizer(derivative, cache_layers, learning_rate = 0.01):

    opt_values = []
    for index, layer in enumerate(layers):
        cache_layers[index] += np.power(derivative[index], 2)
        opt_values.append(-learning_rate * derivative[index] / (np.sqrt(cache_layers[index]) +1e-7))

    return opt_values, cache_layers

def RMSProp_optimizer(derivative, cache_layers, learning_rate = 0.01):
    decay_rate = 0.01
    opt_values = []
    for index, layer in enumerate(layers):
        cache_layers[index] = decay_rate * cache_layers[index] + (1- decay_rate) * np.power(derivative[index], 2)
        opt_values.append(-learning_rate * derivative[index] / (np.sqrt(cache_layers[index]) + 1e-7))

    return opt_values, cache_layers

def update2(opt_values_weight, opt_values_bias, derivative_w, derivative_b):
    for index, layer in enumerate(layers):
        layer.w -= opt_values_weight[index] * derivative_w[index]
        layer.b -= opt_values_bias[index] * derivative_b[index]

def predict():
    test_data, test_labels = load_test_from_file()
    cache = feedforward(test_data[0])
    print('fgh',cache[-1])
    return test_labels, cache[-1]

def compute_accuracy(labels, predicted):
    acc = 0
    for i in range(len(labels)):
        if(np.argmax(predicted[i]) == np.argmax(labels[i])):
            acc +=1

    return acc/len(labels)*100


## methods
def gradient_decsent():

    #learning rate
    lr = 0.5

    epochs = 40
    losses = []
    epochs_num = []

    for i in range(epochs):
        print('epoch number: ', i)
        avg_loss = 0.

        for batch in range(5):
            print('batch number: ', batch+1)
            train_data, train_labels = load_batch_data_from_file(batch+1)

            x = train_data
            y = train_labels

            cache = feedforward(x)
            derivative_w, derivative_b, loss = backprop(y, cache)
            update(lr, derivative_w, derivative_b)

            avg_loss += loss
        
        avg_loss = avg_loss/5
        print('avg ', avg_loss)
        losses.append(avg_loss)
        epochs_num.append(i)

    plot_loss(losses, epochs_num)

def adam(epochs):

    losses = []
    epochs_num = []

    m_weight_layers = []
    v_weight_layers = []

    m_bias_layers = []
    v_bias_layers = []

    for i in range(epochs):
        print('epoch number: ', i)
        avg_loss = 0.

        for batch in range(5):
            print('batch number: ', batch+1)
            train_data, train_labels = load_batch_data_from_file(batch+1)

            x = train_data
            y = train_labels

            cache = feedforward(x)
            derivative_w, derivative_b, loss = backprop(y, cache)

            # initialize
            if(batch == 0):
                for index, layer in enumerate(layers):
                    m_weight_layers.append(np.zeros(derivative_w[index].shape))
                    v_weight_layers.append(np.zeros(derivative_w[index].shape))

                    m_bias_layers.append(np.zeros(derivative_b[index].shape))
                    v_bias_layers.append(np.zeros(derivative_b[index].shape))


            opt_values_weight, m_weight_layers, v_weight_layers = Adam_optimizer(derivative=derivative_w, epoch=i+1, m = m_weight_layers, v = v_weight_layers, learning_rate=0.01)

            opt_values_bias, m_bias_layers, v_bias_layers = Adam_optimizer(derivative=derivative_b, epoch=i+1, m = m_bias_layers, v = v_bias_layers, learning_rate=0.01)

            update2(opt_values_weight, opt_values_bias, derivative_w, derivative_b)

            avg_loss += loss
        
        avg_loss = avg_loss/5
        print('avg ', avg_loss)
        losses.append(avg_loss)
        epochs_num.append(i)

    plot_loss(losses, epochs_num)

def adagrad(epochs):

    losses = []
    epochs_num = []

    cache_weight_layers = []
    cache_bias_layers = []


    for i in range(epochs):
        print('epoch number: ', i)
        avg_loss = 0.

        for batch in range(5):
            print('batch number: ', batch+1)
            train_data, train_labels = load_batch_data_from_file(batch+1)

            x = train_data
            y = train_labels

            cache = feedforward(x)
            derivative_w, derivative_b, loss = backprop(y, cache)

            # initialize
            if(batch == 0):
                for index, layer in enumerate(layers):
                    cache_weight_layers.append(np.zeros(derivative_w[index].shape))
                    cache_bias_layers.append(np.zeros(derivative_b[index].shape))


            opt_values_weight, cache_weight_layers = Adagrad_optimizer(derivative=derivative_w, cache_layers=cache_weight_layers, learning_rate=0.01)

            opt_values_bias, cache_bias_layers = Adagrad_optimizer(derivative=derivative_b, 
            cache_layers=cache_bias_layers, learning_rate=0.01)

            update2(opt_values_weight, opt_values_bias, derivative_w, derivative_b)

            avg_loss += loss
        
        avg_loss = avg_loss/5
        print('avg ', avg_loss)
        losses.append(avg_loss)
        epochs_num.append(i)

    plot_loss(losses, epochs_num)

def RMSProp(epochs):
    losses = []
    epochs_num = []

    cache_weight_layers = []
    cache_bias_layers = []


    for i in range(epochs):
        print('epoch number: ', i)
        avg_loss = 0.

        for batch in range(5):
            print('batch number: ', batch+1)
            train_data, train_labels = load_batch_data_from_file(batch+1)

            x = train_data
            y = train_labels

            cache = feedforward(x)
            derivative_w, derivative_b, loss = backprop(y, cache)

            # initialize
            if(batch == 0):
                for index, layer in enumerate(layers):
                    cache_weight_layers.append(np.zeros(derivative_w[index].shape))
                    cache_bias_layers.append(np.zeros(derivative_b[index].shape))


            opt_values_weight, cache_weight_layers = RMSProp_optimizer(derivative=derivative_w, cache_layers=cache_weight_layers, learning_rate=0.01)

            opt_values_bias, cache_bias_layers = RMSProp_optimizer(derivative=derivative_b, 
            cache_layers=cache_bias_layers, learning_rate=0.01)

            update2(opt_values_weight, opt_values_bias, derivative_w, derivative_b)

            avg_loss += loss
        
        avg_loss = avg_loss/5
        print('avg ', avg_loss)
        losses.append(avg_loss)
        epochs_num.append(i)

    plot_loss(losses, epochs_num)


# make layers
layers = []
layers.append(Layer(input_dimension=1024, output_dimension=512, activation='relu'))
layers.append(Layer(input_dimension=512, output_dimension=512, activation='relu'))
layers.append(Layer(input_dimension=512, output_dimension=512, activation='relu'))
layers.append(Layer(input_dimension=512, output_dimension=512, activation='relu'))
layers.append(Layer(input_dimension=512, output_dimension=128, activation='sigmoid'))
layers.append(Layer(input_dimension=128, output_dimension=10, activation='sigmoid'))


adam(100)


# test_labels, predicted = predict()
# acc = compute_accuracy(test_labels, predicted)
# print('acc ', acc)

def predict2(data):
    cache = feedforward(data)
    return cache[-1].argmax()


def get_acc2(x, y):
    acc = 0
    for xx,yy in zip(x, y):
        s = predict2(xx)
        if s == np.argmax(yy):
            acc +=1
    return acc/len(x)*100

test_data, test_labels = load_test_from_file()
print('acc ', get_acc2(test_data, test_labels))


plt.xlabel("epoch")
plt.ylabel("loss")
plt.title('Adam, 4 layer')
plt.show()