
# Xin: N*Din
# Xout: N*Dout
# W: Din*Dout
# Xout = f(Xin*W)
import numpy as np

class Layer:

    def __init__(self, x_in, w, activation):
        self.x_in = x_in
        self.w = w
        self.activation = activation

    def forward(self):
        x_out = np.dot(self.x_in, self.w)
        for i in np.nditer(x_out, op_flags=['readwrite']):
            i = self.activation_function(i)
        return x_out
        
    def activation_function(self, input):
        if(self.activation == 'ReLU'):
            return self.ReLU(input)
        return 0

    def ReLU(self,x):
        if(x >= 0):
            return x
        return 0

    def derivative_ReLU(self, x):
        if(x >=0 ):
            return 1
        return 0
    