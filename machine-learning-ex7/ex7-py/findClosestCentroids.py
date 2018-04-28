# -*- coding:utf-8 -*-
import numpy as np


def findClosestCentroids(X, center):
    K = np.size(center, 0)
    m = np.size(X, 0)
    idx = np.zeros((m, 1), dtype=int)
    K_temp = np.zeros((K,), dtype=float)
    for i in range(m):
        for j in range(K):
            K_temp[j] = np.sum(np.square(X[i, :] - center[j, :]))
        idx[i] = np.argmin(K_temp)
    return idx
