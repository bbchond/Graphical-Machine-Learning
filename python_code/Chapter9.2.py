import math
import numpy as np
import matplotlib.pyplot as plt

n = 50
x = np.random.randn(n, 2)
y = 2 * (x[:, 0] > x[:, 1]) - 1
b = 5000
a = 50
Y = np.zeros((a, a))
X0 = np.linspace(-3, 3, a)
X = np.empty((50, 50, 2))
X[:, :, 0], X[:, :, 1] = np.meshgrid(X0, X0)

for i in range(b):
    db = math.ceil(2 * np.random.rand() - 1)
    # Bootstrap
    r = np.random.randint(n, size=(n, 1))
    xb = x[r, :].reshape(-1, 2)
    yb = y[r]
    xs, xi = np.sort(xb[:, db]), np.argsort(xb[:, db])
    el = np.cumsum(yb[xi])
    eu = np.cumsum(yb[xi[:: -1]])
    e = eu[-2:: -1] - el[: -1]
    em, ei = np.max(np.abs(e)), np.argmax(np.abs(e))
    c = np.mean(xs[ei: ei + 2])
    s = np.sign(e[ei])
    Y = Y + np.sign(s * (X[:, :, db] - c)) / b

plt.contourf(X0, X0, np.sign(Y))
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()

# 根据书上，每个自助采样获取50个sample，共5000个决策树
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=5000,
    max_samples=50, bootstrap=True, random_state=42
)
bag_clf.fit(x, y)
X_new = np.c_[X[:, :, 0].ravel(), X[:, :, 1].ravel()]
y_pred = bag_clf.predict(X_new).reshape(X[:, :, 0].shape)
plt.contourf(X0, X0, y_pred)
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()
