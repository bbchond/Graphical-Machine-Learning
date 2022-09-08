import numpy as np
import matplotlib.pyplot as plt

n = 50
x = np.c_[np.random.randn(1, 25) - 15, np.random.randn(1, 25) - 5]
x = np.r_[x, np.random.randn(1, n)]
x = np.c_[x.T, np.ones((n, 1))]
x[0: 2, 0] = x[0: 2, 0] + 10
y = np.r_[np.ones((25, 1)), -np.ones((25, 1))]
p = np.random.permutation(n)
x = x[p, :]
y = y[p]

mu = np.zeros((3, 1))
S = np.eye(3)
C = 1
for i in range(len(x)):
    xi = x[i, :].reshape(-1, 1)
    yi = y[i]
    z = S.dot(xi)
    b = xi.T.dot(z) + C
    m = yi * mu.T.dot(xi)
    if m < 1:
        mu = mu + yi * (1 - m) * z / b
        S = S - z.dot(z.T) / b

plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.plot(np.array([-20, 0]), -(mu[2] + np.array([-20, 0]) * mu[0]) / mu[1], 'k-')
plt.axis([-20, 0, -2, 2])
plt.show()
