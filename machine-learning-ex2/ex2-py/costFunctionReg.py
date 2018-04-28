# -*- coding:utf-8 -*-
import numpy as np
from costFunction import costFunction


def costFunctionReg(theta, X, y, l):
    m = len(y)
    J = costFunction(theta, X, y)
    J += l / (2 * m) * ((theta.T).dot(theta) - theta[0] ** 2)
    return J
