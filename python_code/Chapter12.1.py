import numpy as np
import matplotlib.pyplot as plt

n = 100
x = np.concatenate(((np.random.rand(50, 2) - 0.5) * 20, np.random.randn(50, 2)), axis=0)
x[-1, 0] = 14
k = 3
x2 = np.sum(x ** 2, 1).reshape(-1, 1)
RD = np.empty((n, k))
LRD = np.empty((n, k+1))
temp = np.sqrt(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T))
s, t = np.sort(temp, 1), np.argsort(temp, 1)

for i in range(k+1):
    for j in range(k):
        RD[:, j] = np.maximum(s[t[t[:, i], j+1], k], s[t[:, i], j+1])
    LRD[:, i] = 1 / np.mean(RD, 1)
LOF = np.mean(LRD[:, 2: k+1], 1) / LRD[:, 0]

plt.plot(x[:, 0], x[:, 1], 'rx')
# plt.scatter(x[:, 0], x[:, 1], c=None, ms=(LOF * 10), alpha=0.5)
for i in range(n):
    plt.plot(x[i, 0], x[i, 1], 'bo', c=None, ms=(LOF[i] * 10), alpha=0.5)
plt.show()

from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=3)
y_pred = clf.fit(x)
LOF_skl = clf.negative_outlier_factor_
plt.plot(x[:, 0], x[:, 1], 'rx')
for i in range(n):
    plt.plot(x[i, 0], x[i, 1], 'bo', c=None, ms=(-LOF_skl[i] * 10), alpha=0.5)
plt.show()