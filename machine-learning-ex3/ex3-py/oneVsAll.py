# -*- coding:utf-8 -*-
"""
ONEVSALL trains multiple logistic regression classifiers and returns all
the classifiers in a matrix all_theta, where the i-th row of all_theta 
corresponds to the classifier for label i
[all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
logistic regression classifiers and returns each of these classifiers
in a matrix all_theta, where the i-th row of all_theta corresponds 
to the classifier for label i
"""
import numpy as np
import scipy.optimize as scop
from lrCostFunction import lrCostFunction
from gradient import gradient


def oneVsAll(X, y, num_labels, l):
    # X 5000 400 y 5000 1
    m, n = np.shape(X)
    # y = y.reshape((m, 1))
    # 10 401
    all_theta = np.zeros((num_labels, n + 1), dtype=float)
    # 5000 401
    X = np.concatenate([np.ones((m, 1), dtype=float), X], axis=1)

    for i in range(num_labels):
        # 401x1
        initial_theta = np.zeros((n + 1,), dtype=float)
        res = scop.minimize(lrCostFunction, initial_theta, method='BFGS',
                            jac=gradient, args=(X, np.array([int(i) for i in (y == i)]), l), options={'maxiter': 50})
        all_theta[i, :] = res.x
    return all_theta
