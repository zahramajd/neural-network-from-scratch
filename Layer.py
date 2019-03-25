
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
        