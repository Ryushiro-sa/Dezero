if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.core import using_config#動かないから追加


x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
y = F.reshape(x, (6,))  # y = x.reshape(6)
y.backward(retain_grad=True)
print(x.grad)


x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
#y = F.transpose(x)
y = x.transpose()
print(type(y))
#y = x.T
with using_config('enable_backprop', True):#動かないから追加
    y = x.transpose()
y.backward()
print(x.grad)