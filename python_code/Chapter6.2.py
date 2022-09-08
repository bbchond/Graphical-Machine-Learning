import numpy as np
import matplotlib.pyplot as plt

n = 10
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-4, 4, N).T.reshape(N, 1)
y = x + 0.2 * np.random.randn(n, 1)
# 设置离群点
y[n-1] = -4
p = np.empty([n, 2])
p[:, 0] = np.ones((n,))
p[:, 1] = x.reshape(n)
t0 = np.linalg.lstsq(p, y, rcond=-1)[0]
# 阈值设置为1
e = 1
for i in range(1000):
    # 求残差r
    r = np.abs(p.dot(t0) - y)
    # 设置权重矩阵W
    w = np.ones((n, 1))
    w[r > e] = e / r[r > e]
    # w[r <= e] = (1 - r[r <= e] ** 2 / e ** 2) ** 2
    # 根据权重矩阵W更新最小二乘结果t
    t = np.linalg.lstsq(p.T.dot(np.tile(w, (1, 2)) * p), p.T.dot(w * y), rcond=-1)[0]
    if np.linalg.norm(t - t0) < 0.001:
        break
    t0 = t
P = np.empty([N, 2])
P[:, 0] = np.ones((N,))
P[:, 1] = X.reshape(N)
F = P.dot(t0)
plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-4, 4, -4.5, 3.5])
plt.show()
