import numpy as np

from layer import Layer
from softmax import Softmax
from loss import Loss


#TODO:
# layer node 
# softmax node
# loss node

# n = Layer(np.array([[1,2,3],[3,2,1]]),np.array([[3,4,5,6],[3,4,5,6],[3,4,5,6]]),'ReLU')
# o = n.forward()
# print(o)

s = Softmax([1,2,3])
y_p=s.forward()
y=[1,0,0]

l = Loss(y_p= y_p, y=y)

print(l.forward())