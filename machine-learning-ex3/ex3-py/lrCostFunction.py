# -*- coding:utf-8 -*-
"""
LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
regularization
J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
theta as the parameter for regularized logistic regression and the
gradient of the cost w.r.t. to the parameters. 
"""
import numpy as np
from sigmoid import sigmoid


def lrCostFunction(theta, X, y, l):
    m = np.shape(X)[0]
    # y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    s1 = np.log(h).T.dot(y)
    s2 = np.log(1 - h).T.dot(1 - y)
    J = -(s1 + s2) / m + l / (2 * m) * (theta[1:].T.dot(theta[1:]))

    return J
