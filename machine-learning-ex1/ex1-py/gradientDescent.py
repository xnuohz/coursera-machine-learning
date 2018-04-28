# -*- coding:utf-8 -*-
import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    '''
    GRADIENTDESCENT Performs gradient descent to learn theta
    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    taking num_iters gradient steps with learning rate alpha
    '''
    m = len(y)
    y = y.reshape(m, 1)
    J_history = np.zeros((num_iters, 1), dtype='int')
    for itera in range(num_iters):
        temp1 = sum(np.dot(X, theta) - y) / m
        temp2 = np.dot((np.dot(X, theta) - y).T, X[:, 1]) / m
        theta[0] -= alpha * temp1
        theta[1] -= alpha * temp2
        J_history[itera] = computeCost(X, y, theta)
    
    return theta, J_history