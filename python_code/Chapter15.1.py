import numpy as np
import matplotlib.pyplot as plt

n = 200
x = np.c_[np.random.randn(1, 100) - 5, np.random.randn(1, 100) + 5]
x = np.r_[x, np.random.randn(1, n) * 5]
x = np.c_[x.T, np.ones((n, 1))]
y = np.r_[np.ones((100, 1)), -np.ones((100, 1))]
p = np.random.permutation(n)
x = x[p, :]
y = y[p]

t = np.zeros((3, 1))
l = 1
for i in range(len(x)):
    xi = x[i, :].reshape(-1, 1)
    yi = y[i]
    t = t + (yi * np.maximum(0, 1 - t.T.dot(xi).dot(yi)) / (xi.T.dot(xi) + l)) * xi

plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.plot(np.array([-10, 10]), -(t[2] + np.array([-10, 10]) * t[0]) / t[1], 'k-')
plt.axis([-10, 10, -10, 10])
plt.show()
