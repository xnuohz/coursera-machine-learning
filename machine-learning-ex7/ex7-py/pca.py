# -*- coding:utf-8 -*-
import numpy as np
import numpy.linalg as lg


def pca(X):
    m, n = np.shape(X)
    sigma = 1 / m * X.T.dot(X)
    U, S, _ = lg.svd(sigma)
    return U, S