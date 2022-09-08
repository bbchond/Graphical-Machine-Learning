import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls
n = 50
N = 1000
x = np.linspace(-3, 3, n).T
# x = x.reshape(n, -1)
X = np.linspace(-3, 3, N).T
# X = X.reshape(N, -1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x).reshape(n, -1) + np.dot(0.2, np.random.randn(n, 1))
# y = y.reshape(50, -1)

p = np.empty([n, 31])
p[:, 1] = np.ones((n,))
P = np.empty([N, 31])
P[:, 1] = np.ones((N,))

for i in range(1, 16):
    p[:, 2 * i - 1] = np.sin(i / 2 * x)
    p[:, 2 * i] = np.cos(i / 2 * x)
    P[:, 2 * i - 1] = np.sin(i / 2 * X)
    P[:, 2 * i] = np.cos(i / 2 * X)

t1 = nnls(p, y.flatten())[0]
# t = np.linalg.pinv(p).dot(y)
F1 = P.dot(t1)
t2 = nnls((p.dot(np.diagflat(np.append(np.ones((1, 11)), np.zeros((1, 20)))))), y.flatten())[0]
F2 = P.dot(t2)

plt.plot(x, y, 'bo')
plt.plot(X, F1, 'g-', label='LS')
plt.plot(X, F2, 'r--', label='Subspace-Constrained LS')
plt.axis([-2.8, 2.8, -0.8, 1.2])
plt.legend(loc='upper right')
plt.show()