# -*- coding:utf-8 -*-
import numpy as np
from trainLinearReg import trainLinearReg
from linearRegCostFunction import linearRegCostFunction


def validationCurve(x, y, xval, yval):
    lamb_vec = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    err_train = np.zeros((len(lamb_vec,)))
    err_val = np.zeros((len(lamb_vec,)))

    for i in range(len(lamb_vec)):
        lamb = lamb_vec[i]
        theta = trainLinearReg(x, y, lamb)
        err_train[i] = linearRegCostFunction(theta, x, y, 0)
        err_val[i] = linearRegCostFunction(theta, xval, yval, 0)

    return lamb_vec, err_train, err_val
