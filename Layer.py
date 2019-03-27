import numpy as np

# Xin: N*Din
# Xout: N*Dout
# W: Din*Dout
# Xout = f(Xin*W)


class Layer:

    def __init__(self, x_in, w, activation):
        self.x_in = x_in
        self.w = w
        self.activation = activation

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

    def forward(self):
        x_out = np.matmul(self.x_in, self.w)
        for i in np.nditer(x_out, op_flags=['readwrite']):
            i = self.activation_function(i)
        return x_out
        
    def activation_function(self, input):
        if(self.activation == 'ReLU'):
            return self.ReLU(input)
        return 0

    def derivative_activation_function(self, input):
        if(self.activation == 'ReLU'):
            return self.derivative_ReLU(input)
        return 0

    def backward(self,upstream_gradient):
        # upstream_gradient: N*Dout 
        # local_gradient: Din*Dout : Wt
        der_act = np.matmul(self.x_in, self.w)
        for i in np.nditer(der_act, op_flags=['readwrite']):
            i = self.derivative_activation_function(i)

        local_gradient = np.matmul(der_act,self.w)
        gradient =np.matmul(upstream_gradient * local_gradient)
        return gradient