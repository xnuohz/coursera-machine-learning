# -*- coding:utf-8 -*-
import numpy as np


def cofiGradientFunc(X, Theta, y, r, num_users, num_movies, num_features, l):
    X_grad = (r * (X.dot(Theta.T) - y)).dot(Theta) + l * X
    Theta_grad = (r * (X.dot(Theta.T) - y)).T.dot(X) + l * Theta
    return X_grad, Theta_grad
