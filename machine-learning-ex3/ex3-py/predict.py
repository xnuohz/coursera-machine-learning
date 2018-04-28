# -*- coding:utf-8 -*-
"""
PREDICT Predict the label of an input given a trained neural network
p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
trained weights of a neural network (Theta1, Theta2)
"""
import numpy as np
from sigmoid import sigmoid


def predict(theta1, theta2, X):
    m = np.shape(X)[0]
    num_labels = np.shape(theta2)[0]
    # 5000x401
    X = np.concatenate([np.ones((m, 1), dtype=float), X], axis=1)
    # 5000x25
    h = sigmoid(X.dot(theta1.T))
    # 5000x26
    h = np.concatenate([np.ones((m, 1), dtype=float), h], axis=1)
    # 5000x10
    out = sigmoid(h.dot(theta2.T))

    res = np.argmax(out, axis=1) + 1
    return res
