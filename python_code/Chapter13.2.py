import numpy as np
import matplotlib.pyplot as plt

n = 100
x = np.array([2 * np.random.randn(n, 1), np.random.randn(n, 1)]).T.reshape(n, 2)
x = x - x.mean()
U, s, Vt = np.linalg.svd(x.T.dot(x))

plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot([-Vt[0][0] * 9, Vt[0][0] * 9], [-Vt[0][1] * 9, Vt[0][1] * 9], 'k-')
plt.axis([-6, 6, -6, 6])
plt.show()

# 使用sklearn的主成分分析
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
x_sklearn = pca.fit_transform(x)
Vt_sklearn = pca.components_.T[:, 0]
plt.plot(x[:, 0], x[:, 1], 'rx')
plt.plot([-Vt_sklearn[0] * 9, Vt_sklearn[0] * 9], [-Vt_sklearn[1] * 9, Vt_sklearn[1] * 9], 'k-')
plt.axis([-6, 6, -6, 6])
plt.show()
