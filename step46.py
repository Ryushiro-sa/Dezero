if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

from dezero.core import using_config #しゃーない追加

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.SGD(lr).setup(model)

#SGDのloss更新
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    with using_config('enable_backprop', True): #しゃーない追加
        loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(f"SGDの{i}回目のloss: {loss}")

print("-" * 80)


#Momentumのloss更新

lr = 0.2
max_iter = 10000
hidden_size = 10

model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr).setup(model)

for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    with using_config('enable_backprop', True): #しゃーない追加
        loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(f"Momuntumの{i}回目のloss: {loss}")