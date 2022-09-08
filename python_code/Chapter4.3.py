import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-3, 3, N).T.reshape(N, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x).reshape(n, -1) + np.dot(0.2, np.random.randn(n, 1))

x2 = x * x
xx = np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)
hhs = np.asarray([0.03, 0.3, 3])
hhs = 2 * hhs ** 2
ls = np.asarray([0.0001, 0.1, 100])
m = 5
u = []
for i in range(n):
    temp = int(m * (i - 1) / n)
    u.append(temp)
u = np.asarray(u)
u = np.random.permutation(u)

g = np.zeros((3, 3, 5))
# 对hhs，ls数组中的值进行交叉验证
for hk in range(len(hhs)):
    hh = hhs[hk]
    k = np.exp(-xx / hh)
    # 划分4个用于学习(ki,yi==>40*50, 40*1)，1个用于验证(kc, yc==>10*50, 10 * 1)
    for i in range(m):
        ki = k[u != i, :]
        kc = k[u == i, :]
        yi = y[u != i]
        yc = y[u == i]
        for lk in range(len(ls)):
            l = ls[lk]
            t = np.linalg.lstsq((ki.T.dot(ki) + l * np.eye(n)), ki.T.dot(yi), rcond=-1)[0]
            fc = kc.dot(t)
            # 将所有的平方残差存储进数组g
            g[hk, lk, i] = np.mean((fc - yc) ** 2)
# 根据数组g，取能得到最小残差和的h->HH值和l->L值
temp = np.mean(g, 2)
gl = np.min(temp, axis=1)
ggl = np.argmin(temp, 1)

ghl = np.min(gl)
gghl = np.argmin(gl)
L = ls[ggl[gghl]]
HH = hhs[gghl]

K = np.exp(-(np.tile(X ** 2, (1, n)) + np.tile(x2.T, (N, 1)) - 2 * X.dot(x.T)) / HH)
k = np.exp(-xx /HH)
t = np.linalg.lstsq(k.dot(k.T) + L * np.eye(n), k.dot(y), rcond=-1)[0]
# t1 = np.linalg.inv((k.T.dot(k) + L * np.eye(n))).dot(k.T).dot(y)
F = K.dot(t)

plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-2.8, 2.8, -0.7, 1.7])
plt.show()
