import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.linalg import eig


n = 1000
k = 10
a = 3 * np.pi * np.random.rand(n, 1)
x = np.array([a * np.cos(a), 30 * np.random.rand(n, 1), a * np.sin(a)]).T.reshape(-1, 3)
x = x - x.mean()
x2 = (x ** 2).sum(axis=1).reshape(-1, 1)
d = x2 + x2.T - 2 * x.dot(x.T)
p, i = np.sort(d, 0), np.argsort(d, 0)
W = sparse.coo_matrix(d <= np.ones((n, 1)) * p[k, :]).toarray()
W = ((W + W.T) != 0)
D = np.diag(W.sum(axis=1))
L = D - W
res = eig(L, D)
v = np.sort(res[0])[: 3]
v_pos = np.argsort(res[0])[: 3]
t = res[1].T[v_pos]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=40, c=a)
ax.view_init(10, -70)
plt.show()

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.scatter(t[1], t[2], c=a)
plt.show()
plt.show()

"""
尝试使用MDS, LLE, TSNE进行数据可视化操作，并进行效果对比
"""
from sklearn.manifold import MDS
mds = MDS(n_components=2, random_state=42)
x_sklearn = mds.fit_transform(x)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.scatter(x_sklearn[:, 0], x_sklearn[:, 1], c=a)
plt.show()

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_components=2, random_state=42)
x_sklearn = lle.fit_transform(x)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.scatter(x_sklearn[:, 0], x_sklearn[:, 1], c=a)
plt.show()

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
x_sklearn = tsne.fit_transform(x)
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111)
ax.scatter(x_sklearn[:, 0], x_sklearn[:, 1], c=a)
plt.show()
