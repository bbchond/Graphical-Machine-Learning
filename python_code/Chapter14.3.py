import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.linalg import eig

n = 500
c = 2
k = 10
a = np.linspace(0, 2 * np.pi, int(n / 2)).T.reshape(int(n / 2), 1)
t = np.random.permutation(n)

x = np.array([np.array([a * np.cos(a), a * np.sin(a)]).T,
              np.array([(a + np.pi) * np.cos(a), (a + np.pi) * np.sin(a)]).T])
x = x.reshape(n, 2)
x = x + np.random.rand(n, 2)
x = x - x.mean(axis=0)

plt.plot(x[:, 0], x[:, 1], 'bo')
plt.show()

x2 = (x ** 2).sum(axis=1).reshape(n, 1)
d = x2 + x2.T - 2 * x.dot(x.T)
p = np.sort(d, axis=0)
i = np.argsort(d, axis=0)

W = sparse.coo_matrix(d <= np.ones((n, 1)) * p[k, :]).toarray()
W = ((W + W.T) != 0)
D = np.diag(W.sum(axis=1))
L = D - W
res = eig(L, D)
v = np.sort(res[0])[: c-1]
v_pos = np.argsort(res[0])[: c-1]
z = res[1][:, v_pos]

m = z[t[0: c], :]
s = np.empty((c, 1))
s0 = np.empty((c, 1))
s0[: c, 0] = np.inf
z2 = (z ** 2).sum(axis=1)
for o in range(1000):
    m2 = (m ** 2).sum(axis=1).reshape(c, 1)
    temp = m2 + z2.T - 2 * m.dot(z.T)
    u = temp.min(axis=0).reshape(1, -1)
    y = temp.argmin(axis=0).reshape(1, -1)
    for j in range(c):
        m[j, :] = z[np.where(y == j)[1], :].mean(axis=0)
        s[j] = d[np.where(y == j)[1]].mean()
        # print('i is' + str(o) + ', and j is + ' + j)
    if np.linalg.norm(s - s0) < 0.001:
        break
    s0 = s

plt.plot(x[np.where(y == 0)[1], 0], x[np.where(y == 0)[1], 1], 'bo')
plt.plot(x[np.where(y == 1)[1], 0], x[np.where(y == 1)[1], 1], 'rx')
plt.axis([-10, 10, -10, 10])
plt.show()
