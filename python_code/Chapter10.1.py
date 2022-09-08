import math
import numpy as np
import matplotlib.pyplot as plt

n = 90
c = 3
y = np.ones((int(n / c), c))
for i in range(c):
    y[:, i] = y[:, i] * i
y = np.column_stack(y).flatten().reshape(n, 1)

x = np.random.randn(int(n / c), c) + np.tile(np.linspace(-3, 3, c), (int(n / c), 1))
x = np.column_stack(x).flatten().reshape(n, 1)

hh = 2 * 1 ** 2
t0 = np.random.randn(n, c)
for o in range(n * 1000):
    i = math.ceil(n * np.random.rand() - 1)
    yi = int(y[i])
    ki = np.exp(-(x - x[i]) ** 2 / hh)
    ci = np.exp(ki.T.dot(t0))
    t = t0 - 0.1 * (ki.dot(ci)) / (1 + sum(ci))
    t[:, yi] = t[:, yi] + (0.1 * ki).reshape((n, ))
    if np.linalg.norm(t - t0) < 0.000001:
        break
    t0 = t

N = 100
X = np.linspace(-5, 5, N).T.reshape(-1, 1)
K = np.exp(-(np.tile(X ** 2, (1, n)) + np.tile((x ** 2).T, (N, 1)) - 2 * X.dot(x.T)) / hh)
C = np.exp(K.dot(t0))
C = C / np.tile(np.sum(C, 1).reshape(-1, 1), (1, c))
plt.plot(X, C[:, 0], 'b-', label='q(y=1|x)')
plt.plot(X, C[:, 1], 'r--', label='q(y=2|x)')
plt.plot(X, C[:, 2], 'g:', label='q(y=3|x)')
plt.plot(x[np.where(y == 0)[0]], -0.1 * np.ones((int(n / c), 1)), 'bo')
plt.plot(x[np.where(y == 1)[0]], -0.2 * np.ones((int(n / c), 1)), 'rx')
plt.plot(x[np.where(y == 2)[0]], -0.1 * np.ones((int(n / c), 1)), 'gv')
plt.axis([-5, 5, -0.3, 1.8])
plt.legend()
plt.show()


from sklearn.linear_model import LogisticRegression
# sklearn中的逻辑回归默认使用l2惩罚，这会使得曲线光滑，想重现书中例子，我们这里取消了惩罚项，设为none
log_reg = LogisticRegression(penalty='none', solver="lbfgs", random_state=42)
log_reg.fit(x, y.ravel())
y_pred_pro = log_reg.predict_proba(X)
plt.plot(X, y_pred_pro[:, 0], 'b-', label='q(y=1|x)')
plt.plot(X, y_pred_pro[:, 1], 'r--', label='q(y=2|x)')
plt.plot(X, y_pred_pro[:, 2], 'g:', label='q(y=3|x)')
plt.plot(x[np.where(y == 0)[0]], -0.1 * np.ones((int(n / c), 1)), 'bo')
plt.plot(x[np.where(y == 1)[0]], -0.2 * np.ones((int(n / c), 1)), 'rx')
plt.plot(x[np.where(y == 2)[0]], -0.1 * np.ones((int(n / c), 1)), 'gv')
plt.axis([-5, 5, -0.3, 1.8])
plt.legend()
plt.show()