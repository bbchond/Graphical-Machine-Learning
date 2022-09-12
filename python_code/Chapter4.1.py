import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).reshape(-1, 1)
X = np.linspace(-3, 3, N).reshape(-1, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x) + np.dot(0.2, np.random.randn(n, 1))

p = np.empty((n, 31))
p[:, 0] = np.ones((n,))
P = np.empty((N, 31))
P[:, 0] = np.ones((N,))

for i in range(1, 16):
    p[:, 2 * i - 1] = np.sin(i / 2 * x).reshape((n, ))
    p[:, 2 * i] = np.cos(i / 2 * x).reshape((n, ))
    P[:, 2 * i - 1] = np.sin(i / 2 * X).reshape((N, ))
    P[:, 2 * i] = np.cos(i / 2 * X).reshape((N, ))

t1 = np.linalg.lstsq(p, y, rcond=-1)[0]
# t1 = np.linalg.pinv(p).dot(y)
F1 = P.dot(t1)
# 这里只取数据源p的0~10列的数据，11列开始值为0，仅使用前11列与y进行最小二乘
t2 = np.linalg.lstsq((p.dot(np.diagflat(np.append(np.ones((1, 11)), np.zeros((1, 20)))))), y, rcond=-1)[0]
F2 = P.dot(t2)
# 图中可以看到，限制子空间的LS方法减轻了过拟合
plt.plot(x, y, 'bo')
plt.plot(X, F1, 'g-', label='LS')
plt.plot(X, F2, 'r--', label='Subspace-Constrained LS')
plt.axis([-2.8, 2.8, -0.8, 1.2])
plt.legend(loc='upper right')
plt.show()
