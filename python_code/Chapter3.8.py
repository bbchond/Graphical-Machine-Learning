import numpy as np
import math
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-3, 3, N).T.reshape(N, 1)
pix = np.pi * x
y = np.sin(pix) / pix + 0.1 * x + 0.05 * np.random.randn(n, 1)

hh = 2 * 0.3 ** 2
t0 = np.random.randn(n, 1)
learning_rate = 0.1
for o in range(n * 1000):
    # 获取随机样本下标i
    i = math.ceil(np.random.rand() * n - 1)
    # 计算样本x_i的核转换后的结果
    ki = np.exp(-((x - x[i]) * (x - x[i])) / hh)
    # 计算样本i的训练误差梯度
    t = t0 - learning_rate * ki.dot(ki.T.dot(t0) - y[i])
    if np.linalg.norm(t - t0) < 0.000001:
        break
    t0 = t
K = np.exp(-(np.tile(X * X, (1, n)) + np.tile((x * x).T, (N, 1)) - 2 * X.dot(x.T)) / hh)
F = K.dot(t0)

plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-2.8, 2.8, -0.5, 1.2])
plt.show()
