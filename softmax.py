
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
        res = np.apply_along_axis(self.softmax, 1, self.x_in)
        return res

    def backward(self):

        return
        
    def softmax(self,x):
        ex = np.exp(x)
        sum_ex = np.sum(np.exp(x))
        return ex/sum_ex