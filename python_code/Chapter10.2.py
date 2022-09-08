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
x2 = x ** 2
learning_rate = 0.1
N = 100
X = np.linspace(-5, 5, N).T.reshape(-1, 1)
k = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)) / hh)
K = np.exp(-(np.tile(X ** 2, (1, n)) + np.tile(x2.T, (N, 1)) - 2 * X.dot(x.T)) / hh)
Kt = np.empty((N, 3))
for i in range(0, c):
    yk = (y == i)
    ky = k[:, np.where(yk == 1)[0]]
    ty = np.linalg.lstsq((ky.T.dot(ky) + learning_rate * np.eye(np.sum(yk))), ky.T.dot(yk), rcond=-1)[0]
    Kt[:, i] = np.maximum(0, K[:, np.where(yk == 1)[0]].dot(ty)).reshape((-1, ))
ph = Kt / np.tile(np.sum(Kt, 1).reshape(-1, 1), (1, c))
plt.plot(X, ph[:, 0], 'b-', label='q(y=1|x)')
plt.plot(X, ph[:, 1], 'r--', label='q(y=2|x)')
plt.plot(X, ph[:, 2], 'g:', label='q(y=3|x)')
plt.plot(x[np.where(y == 0)[0]], -0.1 * np.ones((int(n / c), 1)), 'bo')
plt.plot(x[np.where(y == 1)[0]], -0.2 * np.ones((int(n / c), 1)), 'rx')
plt.plot(x[np.where(y == 2)[0]], -0.1 * np.ones((int(n / c), 1)), 'gv')
plt.axis([-5, 5, -0.3, 1.8])
plt.legend()
plt.show()


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty='l2', solver="lbfgs", random_state=42)
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