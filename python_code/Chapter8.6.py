import numpy as np
import matplotlib.pyplot as plt

n = 40
x = np.concatenate((np.random.randn(1, int(n / 2)) - 15, np.random.randn(1, int(n / 2)) - 5), axis=1)
x = np.concatenate((x, np.random.randn(1, n)), axis=0).T
y = np.concatenate((np.ones((int(n / 2), 1)), np.ones((int(n / 2), 1)) - 2), axis=0)
# 设置两个离群点
x[0: 2, 0] = x[0: 2, 0] + 60
x = np.concatenate((x, np.ones((n, 1))), axis=1)

learning_rate = 0.01
eta = 0.01
# 初始解设为纯0
t0 = np.zeros((3, 1))
# 反复迭代，计算Ramp损失的最小值
for i in range(1000):
    # 计算边距margin
    m = (x.dot(t0) * y)
    # 在原始的Ramp损失上+m，方便求解其局部最优解(输入值的调整函数v)
    v = m + np.minimum(1, np.maximum(0, 1-m))
    # Ramp损失a
    a = np.abs(v - m)
    # 初始化权重矩阵w，并对a>eta的部分进行更新
    w = np.ones(y.shape)
    w[a > eta] = eta / a[a > eta]
    # 使用矩阵V和矩阵W来计算解t
    t = np.linalg.lstsq((x.T.dot(np.tile(w, (1, 3)) * x) + learning_rate * np.eye(3)), x.T.dot(w * v * y), rcond=-1)[0]
    # 反复执行上述过程，直到解theta达到收敛精度
    if np.linalg.norm(t - t0) < 0.001:
        break
    t0 = t
# 设定值域从-20到50
z = [-20, 50]
plt.plot(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], 'bo')
plt.plot(x[np.where(y == -1)[0], 0], x[np.where(y == -1)[0], 1], 'rx')
plt.plot(z, -(t0[2] + z * t0[0]) / t0[1], 'k-')
# plt.plot(z)
plt.axis([-20, 50, -2, 2])
plt.show()
