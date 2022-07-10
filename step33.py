from dezero.core import using_config

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y
x = Variable(np.array(2.0))
iters = 10
i = 0
for i in range(iters):
    i+= 1
    # print("i: ", i)
    print(i, x)
    with using_config('enable_backprop', True):
        y = f(x)
    # print("y: ", type(y))
    x.cleargrad()
    # print("y.creator: ", type(y.creator))
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
