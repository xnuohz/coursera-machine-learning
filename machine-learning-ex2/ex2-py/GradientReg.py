# -*- coding:utf-8 -*-
from Gradient import Gradient


def GradientReg(theta, X, y, l):
    m, n = X.shape
    theta = theta.reshape(n,)
    grad = Gradient(theta, X, y)
    grad += l / m * theta
    grad[0] -= l / m * theta[0]
    return grad
