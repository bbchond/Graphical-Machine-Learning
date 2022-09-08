import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
n = 50
N = 1000
x = np.linspace(-3, 3, n).T.reshape(n, 1)
X = np.linspace(-3, 3, N).T.reshape(N,1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x).reshape(n, -1) + np.dot(0.2, np.random.randn(n, 1))
x2 = x * x
X2 = X * X
hh = 2 * 0.3 ** 2
learning_rate = 0.1
k = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)) / hh)
K = np.exp(-(np.tile(X2, (1, n)) + np.tile(x2.T, (N, 1)) - 2 * X.dot(x.T)) / hh)

t1 = np.linalg.lstsq(k, y, rcond=-1)[0]
F1 = K.dot(t1)
t2 = np.linalg.lstsq((k.dot(k.T) + learning_rate * np.eye(n)), k.dot(y), rcond=-1)[0]
F2 = K.dot(t2)

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(k, y)
F3 = K.dot(ridge_reg.coef_.reshape(n, 1))

t4 = np.linalg.inv((k.T.dot(k) + learning_rate * np.eye(n))).dot(k.T).dot(y)
F4 = K.dot(t4)
# 可以看到，采用通解求得的theta结果与使用lstsq的结果一致
plt.plot(x, y, 'bo')
plt.plot(X, F1, 'g-')
plt.plot(X, F2, 'r-')
plt.plot(X, F3, 'k-')
plt.plot(X, F4, 'y--')
plt.axis([-2.8, 2.8, -1, 1.5])
plt.show()
