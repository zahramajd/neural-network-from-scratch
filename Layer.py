import numpy as np

# Xin: N*Din
# Xout: N*Dout
# W: Din*Dout
# Xout = f(Xin*W)


class Layer:

    def __init__(self, input_dimension, output_dimension, activation):
        self.activation = activation
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension

        self.w = np.random.randn(input_dimension, output_dimension)
        self.b = np.zeros((1, output_dimension))


    def forward(self, x_in):
        z = np.dot(x_in, self.w) + self.b
        # a = self.sigmoid(z)
        if(self.activation == 'relu'):
            a = self.relu(z)
        if(self.activation == 'sigmoid'):
            a = self.sigmoid(z)
        return a

    def backward(self, a):
        if(self.activation == 'relu'):
            da = self.relu_derv(a)
        if(self.activation == 'sigmoid'):
            da = self.sigmoid_derv(a)

        return da

    def sigmoid(self ,s):
        return 1/(1 + np.exp(-s))

    def sigmoid_derv(self ,s):
        return s * (1 - s)

    def relu(self, s):
        return np.maximum(0,s)

    def relu_derv(self, s):
        ds = np.ones(s.shape)
        ds[s <= 0] = 0
        return ds