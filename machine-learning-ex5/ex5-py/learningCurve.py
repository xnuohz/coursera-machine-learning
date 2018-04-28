# -*- coding:utf-8 -*-
import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def learningCurve(X, y, Xval, yval, l):
    m = np.size(X, 0)

    error_train = np.zeros((m,), dtype=float)
    error_val = np.zeros((m,), dtype=float)

    n = np.size(Xval, 0)
    for i in range(m):
        theta = trainLinearReg(X[0:i + 1, :], y[0:i + 1], l)
        error_train[i] = linearRegCostFunction(
            theta, X[0:i + 1, :], y[0:i + 1], 0)
        error_val[i] = linearRegCostFunction(theta, Xval, yval, 0)

    return error_train, error_val
