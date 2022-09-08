import numpy as np
import matplotlib.pyplot as plt

n = 200
a = np.linspace(0, np.pi, int(n / 2))
u = -10 * np.r_[np.cos(a) + 0.5, np.cos(a) - 0.5].reshape(-1, 1) + np.random.randn(n, 1)
v = 10 * np.r_[np.sin(a), -np.sin(a)].reshape(-1, 1) + np.random.randn(n, 1)
x = np.c_[u, v]
y = np.zeros((n, 1))
y[0] = 1
y[n - 1] = -1
x2 = (x ** 2).sum(axis=1).reshape(-1, 1)
hh = 2 * 1 ** 2
k = np.exp(-(x2 + x2.T - 2 * x.dot(x.T)) / hh)
w = k
t = np.linalg.lstsq((k ** 2) + np.eye(n) + 10 * k.dot(np.diagflat(w.sum(axis=1)) - w).dot(k), k.dot(y), rcond=-1)[0]

m = 100
X = np.linspace(-20, 20, m).reshape(-1, 1)
X2 = (X ** 2)
U = np.exp(-((u ** 2) + X2.T - 2 * u.dot(X.T)) / hh)
V = np.exp(-((v ** 2) + X2.T - 2 * v.dot(X.T)) / hh)
K = V.T.dot(U * t)

plt.contourf(X.reshape((-1, )), X.reshape((-1, )), np.sign(K))
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.plot(x[np.where(y == 0)[0], 0], x[np.where(y == 0)[0], 1], 'k.')
plt.axis([-20, 20, -20, 20])
plt.show()
