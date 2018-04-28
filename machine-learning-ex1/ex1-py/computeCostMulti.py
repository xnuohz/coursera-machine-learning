# -*- coding:utf-8 -*-
import numpy as np

def computeCostMulti(X, y, theta):
    m = len(y)
    return sum(np.power(np.subtract(np.dot(X, theta), y), 2)) / (2 * m)