# -*- coding:utf-8 -*-
import numpy as np


def cofiCostFunc(X, Theta, y, r, num_users, num_movies, num_features, l):
    J = 1 / 2 * np.sum(r * (X.dot(Theta.T) - y)**2) + l / \
        2 * (np.sum(Theta**2) + np.sum(X**2))
    return J
