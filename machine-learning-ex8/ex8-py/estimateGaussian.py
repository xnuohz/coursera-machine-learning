# -*- coding:utf-8 -*-
import numpy as np


def estimateGaussian(X):
    mu = np.mean(X, axis=0)
    sigma2 = np.mean(np.square(X - mu), axis=0)
    return mu, sigma2
