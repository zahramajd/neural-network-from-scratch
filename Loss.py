import numpy as np

# YP: N*10
# Y: N*1
# L=CrossEntropy(YP, Y)


class Loss:

    def __init__(self, Y_hat, Y):
        self.Y_hat = Y_hat
        self.Y = Y

    def forward(self):
        n_samples = self.Y.shape[0]
        logp = - np.log(self.Y_hat[np.arange(n_samples), self.Y.argmax(axis=1)])
        loss = np.sum(logp)/n_samples
        return loss

    def backward(self):
        n_samples = self.Y.shape[0]
        res = self.Y_hat - self.Y
        return res/n_samples
        