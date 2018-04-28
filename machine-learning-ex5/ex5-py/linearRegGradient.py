# -*- coding:utf-8 -*-
import numpy as np


def linearRegGradient(theta, X, y, l):
    m = np.size(X, 0)
    grad = X.T.dot(X.dot(theta) - y) / m + l / m * theta
    grad[0] -= l / m * theta[0]
    return grad
