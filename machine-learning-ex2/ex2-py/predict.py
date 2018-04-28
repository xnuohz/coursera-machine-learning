# -*- coding:utf-8 -*-
"""
PREDICT Predict whether the label is 0 or 1 using learned logistic 
regression parameters theta
p = PREDICT(theta, X) computes the predictions for X using a 
threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)
"""
from sigmoid import sigmoid


def predict(theta, X):
    p = sigmoid(X.dot(theta))
    return p > 0.5
