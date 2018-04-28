# -*- coding:utf-8 -*-
import numpy as np


def projectData(X, U, K):
    """ 映射数据 """
    z = X.dot(U[:, 0:K])
    return z


def recoverData(Z, U, K):
    """ 还原数据 """
    X_rec = np.asmatrix(Z).dot(U[:, 0:K].T)
    return np.asarray(X_rec)
