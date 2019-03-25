import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss


#TODO:
# layer node 
# softmax node
# loss node

n = Layer(np.array([[1,2,3],[3,2,1]]),np.array([[3,4,5,6],[3,4,5,6],[3,4,5,6]]),'ReLU')
o = n.forward()
print(o)

