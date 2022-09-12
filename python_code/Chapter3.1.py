import numpy as np
import matplotlib.pyplot as plt

n = 50
N = 1000
x = np.linspace(-3, 3, n).reshape(-1, 1)
X = np.linspace(-3, 3, N).reshape(-1, 1)
pix = np.pi * x
y = (np.sin(pix) / pix + 0.1 * x) + 0.05 * np.random.randn(n, 1)

p = np.empty((n, 31))
p[:, 0] = np.ones((n,))
P = np.empty((N, 31))
P[:, 0] = np.ones((N,))

for i in range(1, 16):
    p[:, 2 * i - 1] = np.sin(i / 2 * x).reshape((n, ))
    p[:, 2 * i] = np.cos(i / 2 * x).reshape((n, ))
    P[:, 2 * i - 1] = np.sin(i / 2 * X).reshape((N, ))
    P[:, 2 * i] = np.cos(i / 2 * X).reshape((N, ))

t = np.linalg.lstsq(p, y, rcond=-1)[0]
F = P.dot(t)

plt.plot(x, y, 'bo')
plt.plot(X, F, 'g-')
plt.axis([-3.05, 3.05, -0.5, 1.2])
plt.show()

# 两个结果图完全一致，也可以使用scipy.linalg.lstsq()这个函数来实现最小二乘求解
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(p, y)
res = P.dot(lin_reg.coef_.T) + lin_reg.intercept_
plt.plot(x, y, 'bo')
plt.plot(X, res, 'g-')
plt.axis([-3.05, 3.05, -0.5, 1.2])
plt.show()
