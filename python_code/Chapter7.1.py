import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
n = 200
a = np.linspace(0, 4 * np.pi, int(n / 2))
u = np.concatenate((a * np.cos(a), (a + np.pi) * np.cos(a))).T.reshape(n, 1) + np.random.rand(n, 1)
v = np.concatenate((a * np.sin(a), (a + np.pi) * np.sin(a))).T.reshape(n, 1) + np.random.rand(n, 1)
x = np.concatenate((u, v), axis=1)
y = np.concatenate((np.ones((1, int(n / 2))), np.ones((1, int(n / 2))) - 2), axis=1).T.reshape(n, 1)

x2 = np.sum((x ** 2), 1).reshape(-1, 1)
hh = 2 * 1 ** 2
learning_rate = 0.01
k = np.exp(-(np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)) / hh)
t = np.linalg.lstsq((k.dot(k) + learning_rate * np.eye(n)), (k.dot(y)), rcond=-1)[0]

m = 100
X = np.linspace(-15, 15, m).T.reshape(-1, 1)
X2 = X ** 2
U = np.exp(-(np.tile(u ** 2, (1, m)) + np.tile(X2.T, (n, 1)) - 2 * u.dot(X.T)) / hh)
V = np.exp(-(np.tile(v ** 2, (1, m)) + np.tile(X2.T, (n, 1)) - 2 * v.dot(X.T)) / hh)

X_1, X_2 = np.meshgrid(X, X)
plt.contourf(X_1, X_2, np.sign(V.T.dot(U * np.tile(t, (1, m)))))
plt.set_cmap(cmap='Purples')
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-15, 15, -15, 15])
plt.show()
