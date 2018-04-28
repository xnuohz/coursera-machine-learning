# -*- coding:utf-8 -*-
import scipy.optimize as scop
import numpy as np
from linearRegCostFunction import linearRegCostFunction
from linearRegGradient import linearRegGradient


def trainLinearReg(X, y, l):
    # theta参数最好作为cost和gradient函数的第一个参数
    initial_theta = np.zeros((np.size(X, 1), 1), dtype=float)
    theta = scop.fmin_cg(linearRegCostFunction, x0=initial_theta,
                         fprime=linearRegGradient,
                         args=(X, y, l),
                         maxiter=200)
    return theta
