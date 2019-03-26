
# YP: N*10
# Y: N*1
# L=CrossEntropy(YP, Y)
import numpy as np

#TODO:
# backward
class Loss:

    def __init__(self, y, y_p):
        self.y = y
        self.y_p = y_p

    def forward(self):
        return self.cross_entropy(self.y_p, self.y)

    def cross_entropy(self,predictions, targets, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(targets*np.log(predictions+1e-9))/N
        return ce