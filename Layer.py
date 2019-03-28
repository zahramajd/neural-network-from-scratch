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
        # changable variables
        self.w = 0
        self.b = 0


    def ReLU(self,z):
        return np.maximum(0,z)

    def derivative_ReLU(self, dA, z):
        dZ = np.array(dA, copy = True)
        dZ[z <= 0] = 0
        return dZ

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def derivative_sigmoid(self, dA, z):
        sig = self.sigmoid(z)
        return dA * sig * (1 - sig)

    def activation_function(self, input):
        if(self.activation == 'ReLU'):
            return self.ReLU(input)
        if(self.activation == 'sigmoid'):
            return self.sigmoid(input)
        return 0

    def derivative_activation_function(self, input):
        if(self.activation == 'ReLU'):
            return self.derivative_ReLU(input)
        if(self.activation == 'sigmoid'):
            return self.derivative_sigmoid(input)
        return 0

    def forward(self, x_in):       
        z = np.dot(self.w, x_in) + self.b
        x_out = self.activation_function(z)
        return x_out, z

    def backward(self, dx_out, z, x_in):
        m = x_in.shape[1]
        dz = self.derivative_activation_function(dx_out, z)
        dw = np.dot(dz, x_in.T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m
        dx_in = np.dot(self.w.T, dz)

        return dx_in, dw, db
