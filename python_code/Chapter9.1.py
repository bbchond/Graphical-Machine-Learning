import math

import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(50, 2)
# 将y的值转为1(True)或-1(False)
y = 2 * (x[:, 0] > x[:, 1]) - 1
X0 = np.linspace(-3, 3, 50)
X = np.empty((50, 50, 2))
# 根据X0，从坐标向量中返回坐标矩阵
X[:, :, 0], X[:, :, 1] = np.meshgrid(X0, X0)
# 随机索引d
d = math.ceil(2 * np.random.rand() - 1)
# 返回排序后的x的d列，及排序后各个数值对应的原列中的索引位置
xs, xi = np.sort(x[:, d]), np.argsort(x[:, d])
# cumsum函数用来计算累加值，返回的是一个array，各个值为到当前index的值和
el = np.cumsum(y[xi])
# 倒序排列xi
eu = np.cumsum(y[xi[:: -1]])
# eu的倒数第二个到第一个的倒序排列，与el的第一个到倒数第二个，各个位置的差值
e = eu[-2:: -1] - el[: -1]
# 返回e中最大值及索引
em, ei = np.max(np.abs(e)), np.argmax(np.abs(e))

c = np.mean(xs[ei: ei+2])
s = np.sign(e[ei])
Y = np.sign(s * (X[:, :, d] - c))
plt.contourf(X0, X0, Y)
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()
# 直接调用sklearn来执行决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(x, y)
Y = []
for i in range(len(X)):
    y_pred = tree_clf.predict(X[i])
    Y.append(y_pred)
Y = np.asarray(Y)
plt.contourf(X0, X0, Y)
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.axis([-3, 3, -3, 3])
plt.show()
