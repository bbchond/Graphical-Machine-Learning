import numpy as np
import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

n = 50
N = 1000
x = np.linspace(-3, 3, n).reshape(-1, 1)
X = np.linspace(-3, 3, N).reshape(-1, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x) + 0.2 * np.random.randn(n, 1)
x2 = x * x
X2 = X * X
hh = 2 * 0.3 ** 2
learning_rate = 0.1
k = np.exp(-(x2 + x2.T - 2 * x.dot(x.T)) / hh)
K = np.exp(-(X2 + x2.T - 2 * X.dot(x.T)) / hh)

t1 = np.linalg.lstsq(k, y, rcond=-1)[0]
F1 = K.dot(t1)
# Ridge惩罚项
t2 = np.linalg.lstsq((k.dot(k.T) + learning_rate * np.eye(n)), k.dot(y), rcond=-1)[0]
F2 = K.dot(t2)

from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(k, y)
F3 = K.dot(ridge_reg.coef_.reshape(n, 1)) + ridge_reg.intercept_

t4 = np.linalg.pinv((k.T.dot(k) + learning_rate * np.eye(n))).dot(k.T).dot(y)
F4 = K.dot(t4)
# 可以看到，采用通解求得的theta结果与使用lstsq的结果一致
plt.plot(x, y, 'bo')
plt.plot(X, F1, 'g-')
plt.plot(X, F2, 'r-')
plt.plot(X, F3, 'k-')
plt.plot(X, F4, 'y--')
plt.axis([-2.8, 2.8, -1, 1.5])
plt.show()
