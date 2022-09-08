import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt

n = 100
x = np.random.randn(n, 2)
# x[0: int(n / 2), 0] = x[0: int(n / 2), 0] - 4
# x[int(n / 2):, 0] = x[int(n / 2):, 0] + 4
x[0: int(n / 4), 0] = x[0: int(n / 4), 0] - 4
x[int(n / 4): int(n / 2), 0] = x[int(n / 4): int(n / 2), 0] + 4
x = x - x.mean(axis=0)
y = np.r_[np.ones((int(n / 2), 1)), 2 * np.ones((int(n / 2), 1))]

m1 = x[np.where(y == 1)[0], :].mean(axis=0).reshape(1, -1)
m2 = x[np.where(y == 2)[0], :].mean(axis=0).reshape(1, -1)
x1 = x[np.where(y == 1)[0], :] - m1
x2 = x[np.where(y == 2)[0], :] - m2

res = eig(int(n / 2) * m1.T.dot(m1) + int(n / 2) * m2.T.dot(m2), x1.T.dot(x1) + x2.T.dot(x2))
max_index = res[0].argmax()
v = res[0][max_index]
t = res[1][:, max_index]

plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], 'rx')
plt.plot([-t[0] * 99, t[0] * 99], [-t[1] * 99, t[1] * 99], 'k-')
plt.axis([-8, 8, -6, 6])
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components=1)
x = np.random.randn(n, 2)
# x[0: int(n / 2), 0] = x[0: int(n / 2), 0] - 4
# x[int(n / 2):, 0] = x[int(n / 2):, 0] + 4
x[0: int(n / 4), 0] = x[0: int(n / 4), 0] - 4
x[int(n / 4): int(n / 2), 0] = x[int(n / 4): int(n / 2), 0] + 4
lda.fit(x, y.ravel())
x_reduced = lda.transform(x)
plt.axis([-8, 8, -6, 6])
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == 2)[0], 0], x[np.where(y == 2)[0], 1], 'rx')
plt.plot([-lda.scalings_[0] * 99, lda.scalings_[0] * 99],  [-lda.scalings_[1] * 99, lda.scalings_[1] * 99], 'k-')
plt.show()
