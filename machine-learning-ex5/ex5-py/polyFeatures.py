# -*- coding:utf-8 -*-
import numpy as np


def polyFeatures(X, p):
    m = np.size(X, 0)
    X_poly = np.zeros((m, p), dtype=float)
    for i in range(p):
        X_poly[:, i] = np.power(X, i + 1).reshape(-1,)
    return X_poly
