# -*- coding:utf-8 -*-
import numpy as np


def normalEqn(X, y):
    return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
