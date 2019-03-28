import numpy as np

# YP: N*10
# Y: N*1
# L=CrossEntropy(YP, Y)


class Loss:

    def __init__(self, Y_hat, Y):
        self.Y_hat = Y_hat
        self.Y = Y

    def forward(self):
        m = self.Y_hat.shape[1]
        cost = -1 / m * (np.dot(self.Y, np.log(self.Y_hat).T) + np.dot(1 - self.Y, np.log(1 - self.Y_hat).T))
        return np.squeeze(cost)

    def backward(self):
        dx_in = - (np.divide(self.Y, self.Y_hat) - np.divide(1 - self.Y, 1 - self.Y_hat))
        return dx_in