# -*- coding:utf-8 -*-
import numpy as np


def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    x_norm = (X - mu) / sigma
    return x_norm, mu, sigma
