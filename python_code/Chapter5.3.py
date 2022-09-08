import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-3, 3, N).T.reshape(N, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x).reshape(n, -1) + np.dot(0.2, np.random.randn(n, 1))
hh = 2 * 0.3 ** 2
learning_rate = 0.1
t0 = np.random.randn(n, 1)
x2 = x ** 2
k = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)) / hh)
k2 = k.dot(k.T)
ky = k.dot(y)
for i in range(1000):
    t = np.linalg.lstsq(k2 + learning_rate * np.linalg.pinv(np.diagflat(np.abs(t0))), ky, rcond=-1)[0]
    if np.linalg.norm(t - t0) < 0.001:
        break
    t0 = t
K = np.exp(-(np.tile(X ** 2, (1, n)) + np.tile(x2.T, (N, 1)) - 2 * X.dot(x.T)) / hh)
F = K.dot(t0)

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.01)
lasso_reg.fit(k, y)
# print(lasso_reg.coef_)

lasso_pre = K.dot(lasso_reg.coef_) + lasso_reg.intercept_
plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.plot(X, lasso_pre, 'r-')
plt.axis([-2.8, 2.8, -1, 1.5])
plt.show()
