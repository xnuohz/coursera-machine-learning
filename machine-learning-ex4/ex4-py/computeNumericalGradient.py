# -*- coding:utf-8 -*-
import numpy as np


def computeNumericalGradient(J, theta, args):
    numgrad = np.zeros(np.size(theta), dtype=float)
    perturb = np.zeros(np.size(theta), dtype=float)
    epsilon = 1e-4
    for i in range(np.size(theta)):
        perturb[i] = epsilon
        loss1 = J(theta - perturb, *args)
        loss2 = J(theta + perturb, *args)
        numgrad[i] = (loss2 - loss1) / (2 * epsilon)
        perturb[i] = 0
    return numgrad
