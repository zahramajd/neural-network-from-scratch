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
        a = self.sigmoid(z)
        return a

    def backward(self, a):
        return self.sigmoid_derv(a)

    def sigmoid(self ,s):
        return 1/(1 + np.exp(-s))

    def sigmoid_derv(s):
        return s * (1 - s)