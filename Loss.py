
# YP: N*1
# Y: N*1 L=CrossEntropy(YP, Y)
import numpy as np
class Loss:

    def __init__(self, y, y_p):
        self.y = y
        self.y_p = y_p