import numpy as np

# Xin: N*10
# Xout: N*10
# Xout = softmax(Xin)

#TODO:
# backward

class Softmax:

    def forward(self, x_in):
        x_out = np.apply_along_axis(self.softmax, 1, x_in)
        return x_out


