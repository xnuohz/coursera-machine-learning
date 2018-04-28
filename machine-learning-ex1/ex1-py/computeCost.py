# -*- coding:utf-8 -*-
import numpy as np


def computeCost(X, y, theta):
    '''
    COMPUTECOST Compute cost for linear regression
    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    '''
    m = len(y)
    return np.sum(np.square(X.dot(theta) - y)) / (2 * m)
