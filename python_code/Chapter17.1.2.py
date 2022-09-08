import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

n = 100
x = np.random.randn(n, 2)
# x[0: int(n / 2), 0] = x[0: int(n / 2), 0] - 4
# x[int(n / 2):, 0] = x[int(n / 2):, 0] + 4
x[0: int(n / 4), 0] = x[0: int(n / 4), 0] - 4
x[int(n / 4): int(n / 2), 0] = x[int(n / 4): int(n / 2), 0] + 4
x = x - x.mean(axis=0)
y = np.r_[np.ones((int(n / 2), 1)), 2 * np.ones((int(n / 2), 1))]

Sw = np.zeros((2, 2))
Sb = np.zeros((2, 2))
for j in (1, 2):
    p = x[np.where(y == j)[0], :]
    p1 = p.sum(axis=0).reshape(1, -1)
    p2 = (p ** 2).sum(axis=1).reshape(-1, 1)
    nj = (y == j).sum()
    W = np.exp(-(p2 + p2.T - 2 * p.dot(p.T)))
    G = p.T.dot(W.sum(axis=1).reshape(-1, 1) * p) - p.T.dot(W).dot(p)
    Sb = Sb + G / n + p.T.dot(p) * (1 - nj / n) + p1.T.dot(p1) / n
    Sw = Sw + G / nj

res = eig((Sb + Sb.T) / 2, (Sw + Sw.T) / 2)
max_index = res[0].argmax()
v = res[0][max_index]
t = res[1][:, max_index]

plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], 'rx')
plt.plot([-t[0] * 99, t[0] * 99], [-t[1] * 99, t[1] * 99], 'k-')
plt.axis([-8, 8, -6, 6])
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
y_pred = kmeans.fit_transform(x)
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], 'rx')
plt.plot(kmeans.cluster_centers_[:, 0] * 99, kmeans.cluster_centers_[:, 1] * 99, 'k-')
plt.axis([-8, 8, -6, 6])
plt.show()
