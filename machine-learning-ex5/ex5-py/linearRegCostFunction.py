# -*- coding:utf-8 -*-
import numpy as np


def linearRegCostFunction(theta, X, y, l):
    m = np.size(X, 0)
    J = np.sum(np.square(X.dot(theta) - y)) / (2 * m) + l / \
        (2 * m) * (theta.T.dot(theta) - theta[0] ** 2)
    return J
