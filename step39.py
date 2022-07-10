if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

from dezero.core import using_config #しゃーない追加

x = Variable(np.array([1, 2, 3, 4, 5, 6]))
y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
with using_config('enable_backprop', True): #しゃーない追加
    y = F.sum(x)
y.backward()
print(y)
print(x.grad)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
with using_config('enable_backprop', True): #しゃーない追加
    y = F.sum(x, axis=0)
y.backward()
print(y)
print(x.grad)

x = Variable(np.random.randn(2, 3, 4, 5))
with using_config('enable_backprop', True): #しゃーない追加
    y = x.sum(keepdims=True)
print(y.shape)