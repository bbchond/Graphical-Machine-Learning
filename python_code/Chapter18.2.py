import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 200
x = np.random.randn(n, 1)
y = np.random.randn(n, 1) + 1
hhs = 2 * np.array([0.5, 1, 3]) ** 2
ls = 10. ** np.array([-2, -1, 0])
m = 5
x2 = (x ** 2).reshape(-1, 1)
y2 = (y ** 2).reshape(-1, 1)
xx = x2 + x2.T - 2 * x.dot(x.T)
yx = y2 + x2.T - 2 * y.dot(x.T)

u = np.floor(m * np.arange(0, n) / n)
u = u[np.random.permutation(n)]
v = np.floor(m * np.arange(0, n) / n)
v = v[np.random.permutation(n)]

g = np.zeros((len(hhs), len(ls), m))
for hk in range(len(hhs)):
    hh = hhs[hk]
    k = np.exp(-xx / hh)
    r = np.exp(-yx / hh)
    U = (np.pi * hh / 2) ** (1 / 2) * np.exp(-xx / (2 * hh))
    for i in range(m):
        vh = (k[u != i, :].mean(axis=0) - r[v != i, :].mean(axis=0)).reshape(-1, 1)
        z = (k[u == i, :].mean(axis=0) - r[v == i, :].mean(axis=0)).reshape(1, -1)
        for lk in range(len(ls)):
            l = ls[lk]
            a = np.linalg.inv(U + l * np.eye(n)).dot(vh)
            g[hk, lk, i] = a.T.dot(U).dot(a) - 2 * z.dot(a)

temp = g.mean(axis=2)
gl, ggl = temp.min(axis=1), temp.argmin(axis=1)
ghl, gghl = gl.min(), gl.argmin()
L = ls[ggl[gghl]]
HH = hhs[gghl]
k = np.exp(-xx / HH)
r = np.exp(-yx / HH)
vh = k.mean(axis=0) - r.mean(axis=0)
vh = vh.reshape(-1, 1)
U = (np.pi * HH / 2) ** (1 / 2) * np.exp(-xx / (2 * HH))
a = np.linalg.inv(U + L * np.eye(n)).dot(vh)
s = (np.r_[k, r]).dot(a)
L2 = 2 * a.T.dot(vh) - a.T.dot(U).dot(a)

plt.plot(np.r_[x, y], s, 'rx')
plt.show()
