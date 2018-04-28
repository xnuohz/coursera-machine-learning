# -*- coding:utf-8 -*-
'''
COSTFUNCTION Compute cost and gradient for logistic regression
J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
parameter for logistic regression and the gradient of the cost
w.r.t. to the parameters.
'''
import numpy as np
from sigmoid import sigmoid


def Gradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    grad = ((X.T).dot(sigmoid(np.dot(X, theta)) - y)) / m
    return grad.flatten()
