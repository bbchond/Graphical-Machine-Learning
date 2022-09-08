import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-3, 3, N).T.reshape(N, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x).reshape(n, -1) + 0.2 * np.random.randn(n, 1)
y[int(n / 2)] = -0.5
hh = 2 * 0.3 ** 2
learning_rate = 0.1
eta = 0.1
x2 = x ** 2
k = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)) / hh)
t0 = np.random.randn(n, 1)  # 随机初始化
# t0 = np.linalg.lstsq(k, y, rcond=-1)[0]  # 使用最小二乘解初始化
for i in range(1000):
    # 计算残差r
    r = np.abs(k.dot(t0) - y)
    # 初始化权重矩阵W
    w = np.ones((n, 1))
    # 更新W
    w[r > eta] = eta / r[r > eta]

    Z = k.dot(np.tile(w, (1, n)) * k + learning_rate * np.linalg.pinv(np.diagflat(np.abs(t0))))
    t = np.linalg.lstsq(Z + 0.000001 * np.eye(n), k.dot(w * y), rcond=-1)[0]
    if np.linalg.norm(t - t0) < 0.001:
        break
    t0 = t
K = np.exp(-(np.tile(X ** 2, (1, n)) + np.tile(x2.T, (N, 1)) - 2 * X.dot(x.T)) / hh)
F = K.dot(t0)
plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-2.8, 2.8, -1, 1.5])
plt.show()
