import numpy as np
import math
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).reshape(-1, 1)
X = np.linspace(-3, 3, N).reshape(-1, 1)
pix = np.pi * x
y = np.sin(pix) / pix + 0.1 * x + 0.05 * np.random.randn(n, 1)

hh = 2 * 0.3 ** 2
t0 = np.random.randn(n, 1)
learning_rate = 0.1
for o in range(n * 1000):
    # 获取随机样本下标i
    i = math.ceil(np.random.rand() * n - 1)
    # 计算样本x_i的核转换后的结果
    ki = np.exp(-((x - x[i]) * (x - x[i])) / hh)
    # 计算样本i的训练误差梯度
    t = t0 - learning_rate * ki.dot(ki.T.dot(t0) - y[i])
    if np.linalg.norm(t - t0) < 0.000001:
        break
    t0 = t
K = np.exp(-((X ** 2) + (x ** 2).T - 2 * X.dot(x.T)) / hh)
F = K.dot(t0)

plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-2.8, 2.8, -0.5, 1.2])
plt.show()

from sklearn.linear_model import SGDRegressor

k_x = np.exp(-((x ** 2) + (x ** 2).T - 2 * x.dot(x.T)) / hh)
sgd_reg = SGDRegressor(max_iter=n * 1000, tol=0.000001, penalty=None, eta0=0.001, random_state=42)
sgd_reg.fit(k_x, y.reshape((n, )))
res = K.dot(sgd_reg.coef_.T) + sgd_reg.intercept_
plt.plot(x, y, 'bo')
plt.plot(X, res, 'g-')
plt.axis([-3.05, 3.05, -0.5, 1.2])
plt.show()

# 我们可以很明显看到，常用的线性回归(即通用最小二乘严重过拟合)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(k_x, y)
res = K.dot(lin_reg.coef_.T) + lin_reg.intercept_
plt.plot(x, y, 'bo')
plt.plot(X, res, 'g-')
plt.axis([-3.05, 3.05, -0.5, 1.2])
plt.show()