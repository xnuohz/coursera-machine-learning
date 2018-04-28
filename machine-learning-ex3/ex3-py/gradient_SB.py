# -*- coding:utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def gradient_SB(theta, X, y, l):
    m = np.shape(X)[0]
    h = sigmoid(X.dot(theta))
    grad = X.T.dot(h - y) / m + l / m * \
        np.concatenate(([[0]], theta[1:].reshape(-1, 1)))
    return grad
