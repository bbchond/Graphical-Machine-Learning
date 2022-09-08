import numpy as np
import matplotlib.pyplot as plt

n = 500
a = np.linspace(0, 2 * np.pi, int(n / 2)).T.reshape(int(n / 2), 1)
t = np.random.permutation(n)

x = np.array([np.array([a * np.cos(a), a * np.sin(a)]).T,
              np.array([(a + np.pi) * np.cos(a), (a + np.pi) * np.sin(a)]).T])
x = x.reshape(n, 2)
x = x + np.random.rand(n, 2)
x = x - x.mean(axis=0)
x2 = (x ** 2).sum(axis=1).reshape(n, 1)
y = np.array([np.ones((1, int(n / 2))), np.zeros((1, int(n / 2)))]).reshape(-1)
d = x2 + x2.T - 2 * x.dot(x.T)

hhs = 2 * np.array([0.5, 1, 2]) ** 2
ls = np.power(10., np.array([-5, -4, -3]))
m = 5

u = []
for i in range(n):
    temp = int(m * (i - 1) / n)
    u.append(temp)
u = np.asarray(u)
u = np.random.permutation(u)

g = np.zeros((len(hhs), len(ls), m))
for hk in range(len(hhs)):
    hh = hhs[hk]
    k = np.exp(-d / hh)
    for j in np.unique(y):
        for i in range(m):
            ki = k[u != i]
            ki = ki[:, np.where(y == j)[0]]
            kc = k[u == i]
            kc = kc[:, np.where(y == j)[0]]
            Gi = ki.T.dot(ki) * (((u != i) & (y == j)).sum()) / ((u != i).sum() ** 2)
            Gc = kc.T.dot(kc) * (((u == i) & (y == j)).sum()) / ((u == i).sum() ** 2)
            hi = k[(u != i) & (y == j)][:, y == j].sum(axis=0).reshape(-1, 1) / ((u != i).sum() ** 2)
            hc = k[(u == i) & (y == j)][:, y == j].sum(axis=0).reshape(-1, 1) / ((u == i).sum() ** 2)
            for lk in range(len(ls)):
                l = ls[lk]
                a = np.linalg.lstsq((Gi + l * np.eye((y == j).sum())), hi, rcond=-1)[0]
                g[hk, lk, i] = g[hk, lk, i] + a.T.dot(Gc).dot(a) / 2 - hc.T.dot(a)

g = g.mean(axis=2)
gl, ggl = g.min(axis=1), g.argmin(axis=1)
ghl, gghl = gl.min(), gl.argmin()
L = ls[ggl[gghl]]
HH = hhs[gghl]
s = -1/2
res = []
for j in np.unique(y):
    k = np.exp(-d[:, y == j] / HH)
    h = k[y == j, :].sum(axis=1) / n
    t = (y == j).sum()
    s = s + h.reshape(1, -1).dot(np.linalg.lstsq(k.T.dot(k) * t / (n ** 2) + L * np.eye(t), h, rcond=-1)[0] / 2)
    res.append(s)
print('Information={}'.format(s))
