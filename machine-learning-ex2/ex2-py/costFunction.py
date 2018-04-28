# -*- coding:utf-8 -*-
'''
COSTFUNCTION Compute cost and gradient for logistic regression
J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
parameter for logistic regression and the gradient of the cost
w.r.t. to the parameters.
'''
import numpy as np
from sigmoid import sigmoid


def costFunction(theta, X, y):
    m = len(y)
    y = y.reshape((m, 1))
    s1 = np.log(sigmoid(np.dot(X, theta))).reshape((m, 1))
    s2 = np.log(1 - sigmoid(np.dot(X, theta))).reshape((m, 1))
    s = y * s1 + (1 - y) * s2
    return -np.sum(s) / m
