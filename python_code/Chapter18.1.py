import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

n = 300
x = np.random.randn(n, 1)
y = np.random.randn(n, 1) + 0.5
hhs = 2 * np.array([1, 5, 10]) ** 2
ls = 10. ** np.array([-3, -2, -1])
m = 5
b = 0.5
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
    for i in range(m):
        ki = k[u != i, :]
        ri = r[v != i, :]
        h = ki.mean(axis=0).reshape(-1, 1)
        kc = k[u == i, :]
        rj = r[v == i, :]
        G = b * ki.T.dot(ki) / (u != i).sum() + (1 - b) * ri.T.dot(ri) / (v != i).sum()
        for lk in range(len(ls)):
            l = ls[lk]
            a = np.linalg.inv(G + l * np.eye(n)).dot(h)
            kca = kc.dot(a)
            g[hk, lk, i] = b * (kca ** 2).mean() + (1 - b) * ((rj.dot(a)) ** 2).mean()
            g[hk, lk, i] = g[hk, lk, i] / 2 - kca.mean(axis=0)

temp = g.mean(axis=2)
gl, ggl = temp.min(axis=1), temp.argmin(axis=1)
ghl, gghl = gl.min(), gl.argmin()
L = ls[ggl[gghl]]
HH = hhs[gghl]
k = np.exp(-xx / HH)
r = np.exp(-yx / HH)
# s = r.dot(np.linalg.lstsq((b * k.T.dot(k) / n + (1 - b) * r.T.dot(r) / n + L * np.eye(n)),
#                           k.mean(axis=0).reshape(-1, 1), rcond=-1)[0])
s = r.dot(np.linalg.inv(b * k.T.dot(k) / n + (1 - b) * r.T.dot(r) / n + L * np.eye(n))
          .dot(k.mean(axis=0).reshape(-1, 1)))

x_pdf, y_pdf = norm.pdf(x), norm.pdf(y)
plt.scatter(x, x_pdf)
plt.scatter(y, y_pdf)
plt.plot(y, s, 'rx')
plt.show()
