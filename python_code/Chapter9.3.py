import math
import numpy as np
import matplotlib.pyplot as plt

n = 50
x = np.random.randn(n, 2)
y = 2 * (x[:, 0] > x[:, 1]) - 1
b = 5000
a = 50
Y = np.zeros((a, a))
yy = np.zeros(y.shape)
w = np.ones((n, 1)) / n

X0 = np.linspace(-3, 3, a)
X = np.empty((50, 50, 2))
X[:, :, 0], X[:, :, 1] = np.meshgrid(X0, X0)

for i in range(b):
    wy = w * y
    d = math.ceil(2 * np.random.rand() - 1)
    xs, xi = np.sort(x[:, d]), np.argsort(x[:, d])
    el = np.cumsum(wy[xi])
    eu = np.cumsum(wy[xi[:: -1]])
    e = eu[-2:: -1] - el[: -1]
    em, ei = np.max(np.abs(e)), np.argmax(np.abs(e))
    c = np.mean(xs[ei: ei + 2])
    s = np.sign(e[ei])
    yh = np.sign(s * (x[:, d] - c))
    R = w.T.dot(1 - yh * y) / 2
    t = np.log((1 - R) / R) / 2
    yy = yy + yh * t
    w = np.exp(-yy * y)
    w = w / np.sum(w)
    Y = Y + np.sign(s * (X[:, :, d] - c)) * t

plt.contourf(X0, X0, np.sign(Y))
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=5000,
    algorithm='SAMME.R', learning_rate=0.5
)
ada_clf.fit(x, y)
X_new = np.c_[X[:, :, 0].ravel(), X[:, :, 1].ravel()]
y_pred = ada_clf.predict(X_new).reshape(X[:, :, 0].shape)
plt.contourf(X0, X0, y_pred)
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()
