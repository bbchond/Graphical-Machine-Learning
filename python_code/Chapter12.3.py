import numpy as np
import matplotlib.pyplot as plt


def kliep(k, r):
    a0 = np.random.rand(k.shape[1], 1)
    b = np.mean(r, 0).reshape(-1, 1)
    c = sum(b ** 2)
    for o in range(1000):
        a = a0 + 0.01 * k.T.dot((1 / k).dot(a0))
        a = a + b * (1 - sum(b * a)) / c
        a = np.maximum(0, a)
        a = a / sum(b * a)
        if np.linalg.norm(a - a0) < 0.001:
            break
        a0 = a
    return a0


if __name__ == '__main__':
    n = 100
    x = np.random.randn(n, 1)
    y = np.random.randn(n, 1)
    y[n - 1] = 5
    hhs = 2 * np.array([1, 5, 10]) ** 2
    m = 5
    x2 = x ** 2
    y2 = y ** 2
    xx = np.tile(x2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * x.dot(x.T)
    yx = np.tile(y2, (1, n)) + np.tile(x2.T, (n, 1)) - 2 * y.dot(x.T)

    u = []
    for i in range(n):
        temp = int(m * (i - 1) / n)
        u.append(temp)
    u = np.asarray(u)
    u = np.random.permutation(u)

    g = np.zeros((3, 5))

    for hk in range(len(hhs)):
        hh = hhs[hk]
        k = np.exp(-xx / hh)
        r = np.exp(-yx / hh)
        for i in range(m):
            g[hk, i] = np.mean(k[u == i, :].dot(kliep(k[u != i, :], r)))
    temp = g.mean(axis=1)
    gh, ggh = temp.max(), temp.argmax()
    HH = hhs[ggh]
    k = np.exp(-xx / HH)
    r = np.exp(-yx / HH)
    s = r.dot(kliep(k, r))
    plt.plot(y, s, 'rx')
    plt.show()
