
# Xin: N*D
# Xout: N*1
# Xout = softmax(Xin)
import numpy as np
class Softmax:

    def __init__(self, x_in):
        self.x_in = x_in
        