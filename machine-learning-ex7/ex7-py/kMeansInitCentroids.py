# -*- coding:utf-8 -*-
import numpy as np


def kMeansInitCentroids(X, K):
    randix = np.random.permutation(np.size(X, 0))
    centroids = X[randix[:K], :]
    return centroids
