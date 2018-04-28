# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from polyFeatures import polyFeatures


def plotFit(min_x, max_x, mu, sigma, theta, p):
    X = np.arange(min_x - 15, max_x + 25, 0.05)
    X_poly = polyFeatures(X, p)
    X_poly = (X_poly - mu) / sigma
    m = np.size(X_poly, 0)
    X_poly = np.concatenate(
        (np.ones((m, 1), dtype=float), X_poly), axis=1)
    return X, X_poly.dot(theta)
