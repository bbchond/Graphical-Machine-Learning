import numpy as np
import matplotlib.pyplot as plt

n = 300
c = 3
t = np.random.permutation(n)
x = np.array([[np.random.randn(1, 100) - 2, np.random.randn(1, 100), np.random.randn(1, 100) + 2],
             [np.random.randn(1, 100), np.random.randn(1, 100) + 4, np.random.randn(1, 100)]]).T.reshape(n, 2)
# 随机选择3个簇中心
m = x[t[0: c], :]
x2 = (x ** 2).sum(axis=1).reshape(n, 1)
s = np.empty((c, 1))
s0 = np.empty((c, 1))
s0[: c, 0] = np.inf

for i in range(1000):
    m2 = (m ** 2).sum(axis=1).reshape(c, 1)
    temp = m2 + x2.T - 2 * m.dot(x.T)
    d = temp.min(axis=0).reshape(1, -1)
    y = temp.argmin(axis=0).reshape(1, -1)
    for j in range(c):
        # update the center of cluster
        m[j, :] = x[np.where(y == j)[1], :].mean(axis=0)
        # update the total distance for point x to center{m}
        s[j] = d[y == j].mean()
    if np.linalg.norm(s - s0) < 0.001:
        break
    s0 = s
# 这里，还需要仔细思考为什么np.where(y==0)的返回tuple组成与之前不一样，也许之前是(n, 1)而这里是(1, n)??
plt.plot(x[np.where(y == 0)[1], 0], x[np.where(y == 0)[1], 1], 'bo')
plt.plot(x[np.where(y == 1)[1], 0], x[np.where(y == 1)[1], 1], 'rx')
plt.plot(x[np.where(y == 2)[1], 0], x[np.where(y == 2)[1], 1], 'gv')
plt.title('use method of book')
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=c)
y_pred = kmeans.fit_predict(x)
plt.plot(x[np.where(y_pred == 0)[0], 0], x[np.where(y_pred == 0)[0], 1], 'bo')
plt.plot(x[np.where(y_pred == 1)[0], 0], x[np.where(y_pred == 1)[0], 1], 'rx')
plt.plot(x[np.where(y_pred == 2)[0], 0], x[np.where(y_pred == 2)[0], 1], 'gv')
plt.title('use method of sklearn')
plt.show()
