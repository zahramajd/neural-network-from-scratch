
# Xin: N*10
# Xout: N*10
# Xout = softmax(Xin)
import numpy as np

#TODO:
# backward
class Softmax:

    def __init__(self, x_in):
        self.x_in = x_in

    def forward(self):
        ex = np.exp(self.x_in)
        sum_ex = np.sum(np.exp(self.x_in))
        return ex/sum_ex

    def backward(self):

        return
        