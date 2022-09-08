import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

n = 100
x = np.array([2 * np.random.randn(n, 1), np.random.randn(n, 1)]).T.reshape(n, 2)
x = x - x.mean()
x2 = (x ** 2).sum(axis=1).reshape(-1, 1)

# W = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)))
W = np.exp(-(x2 + x2.T - 2 * x.dot(x.T)))
D = np.diag(W.sum(axis=1))
L = D - W
z = x.T.dot(D).dot(x)
z = (z + z.T) / 2
res = eig(x.T.dot(L).dot(x), z)
v = res[0].min()
v_pos = res[0].argmin()
t = res[1][v_pos]

plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot([-t[0] * 9, t[0] * 9], [-t[1] * 9, t[1] * 9])
plt.axis([-6, 6, -6, 6])
plt.show()

# TODO 这里，尝试自己实现一个LPP(局部保持投影)的方法，参考网址: https://blog.csdn.net/qq_33764934/article/details/103370588
