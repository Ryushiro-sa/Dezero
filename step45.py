if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Model
from dezero.core import using_config
import dezero.layers as L
import dezero.functions as F
from dezero import Layer


##パラメータにアクセスする様子を確認
model = Layer()
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)


def predict(x):
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y

#全てのパラメータにアクセスする様子
for p in model.params():
    print(f"params:{p}")

#全てのパラメータの勾配リセット
model.cleargrads()



##モデルを使って問題を解く
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# ハイパーパラメータの設定
lr = 0.2
max_iter = 10000
hidden_size = 10

# モデル定義
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)

#学習開始
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    with using_config('enable_backprop', True): #しゃーない追加
        loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data
    if i % 1000 == 0:
        print(f"{i}回目のloss: {loss}")