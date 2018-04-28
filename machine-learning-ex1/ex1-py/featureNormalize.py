# -*- coding:utf-8 -*-
'''
FEATURENORMALIZE Normalizes the features in X 
FEATURENORMALIZE(X) returns a normalized version of X where
the mean value of each feature is 0 and the standard deviation
is 1. This is often a good preprocessing step to do when
working with learning algorithms.
'''
import numpy as np
def featureNormalize(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    m = np.shape(X)[0]
    for i in range(m):
        X_norm[i] = np.divide(X_norm[i] - mu, sigma)
    return X_norm, mu, sigma