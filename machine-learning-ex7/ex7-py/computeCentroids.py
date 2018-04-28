# -*- coding:utf-8 -*-
import numpy as np


def computeCentroids(X, idx, K):
    m, n = np.shape(X)
    centroids = np.zeros((K, n), dtype=float)
    for i in range(K):
        centroids[i, :] = (X.T.dot(idx == i) / np.sum(idx == i)).reshape(-1)
    return centroids
