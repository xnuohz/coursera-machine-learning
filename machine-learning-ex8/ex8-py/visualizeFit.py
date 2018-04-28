# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from multivariateGaussian import multivariateGaussian


def visualizeFit(X, mu, sigma2):
    grid = np.arange(0, 35.5, 0.5)
    x1, x2 = np.meshgrid(grid, grid)

    # flatten(): 横轴方向降   flatten('F'): 竖轴方向降
    # 1 2 => 1 2 3 4         1 2 => 1 3 2 4
    # 3 4                    3 4
    Z = multivariateGaussian(
        np.c_[x1.flatten('F'), x2.flatten('F')], mu, sigma2)
    Z = Z.reshape(x1.shape, order='F')

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], marker='x', c='b', s=15, linewidth=1)

    # Do not plot if there are infinities
    if np.sum(np.isinf(X)) == 0:
        lvls = 10 ** np.arange(-20, 0, 3).astype(np.float)
    plt.contour(x1, x2, Z, levels=lvls, colors='r', linewidths=0.7)
