# -*- coding:utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def gradient(theta, X, y, l):
    m = np.shape(X)[0]
    h = sigmoid(X.dot(theta))
    grad = np.zeros(np.size(theta))
    grad[0] = 1 / m * (X[:, 0].dot(h - y))
    grad[1:] = 1 / m * (X[:, 1:].T.dot(h - y)) + l / m * theta[1:]
    return grad
