# -*- coding:utf-8 -*-
import numpy as np
from computeCostMulti import computeCostMulti

def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = len(y)
    y = y.reshape(m, 1)
    J_history = np.zeros((num_iters, 1), dtype='float')
    for itera in range(num_iters):
        temp1 = sum(np.dot(X, theta) - y) / m
        temp2 = np.dot((np.dot(X, theta) - y).T, X[:, 1]) / m
        temp3 = np.dot((np.dot(X, theta) - y).T, X[:, 2]) / m
        theta[0] -= alpha * temp1
        theta[1] -= alpha * temp2
        theta[2] -= alpha * temp3
        J_history[itera] = computeCostMulti(X, y, theta)

    return theta, J_history