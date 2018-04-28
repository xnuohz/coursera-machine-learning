# -*- coding:utf-8 -*-
import numpy as np
from sigmoid import sigmoid


def predict(theta1, theta2, X):
    m = np.size(X, 0)
    X = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1)
    temp1 = sigmoid(X.dot(theta1.T))
    temp = np.concatenate((np.ones((m, 1), dtype=float), temp1), axis=1)
    temp2 = sigmoid(temp.dot(theta2.T))
    p = np.argmax(temp2, axis=1) + 1
    return p
