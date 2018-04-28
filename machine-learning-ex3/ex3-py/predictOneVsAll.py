# -*- coding:utf-8 -*-
"""
PREDICT Predict the label for a trained one-vs-all classifier. The labels 
are in the range 1..K, where K = size(all_theta, 1). 
p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
for each example in the matrix X. Note that X contains the examples in
rows. all_theta is a matrix where the i-th row is a trained logistic
regression theta vector for the i-th class. You should set p to a vector
of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
for 4 examples) 
"""
import numpy as np
from sigmoid import sigmoid


def predictOneVsAll(all_theta, X):
    m, _ = np.shape(X)
    num_labels, _ = np.shape(all_theta)
    # X 5000 401 all_theta 10 401
    X = np.concatenate([np.ones((m, 1), dtype=float), X], axis=1)
    # 每行最大值对应下标
    pred = np.argmax(X.dot(all_theta.T), axis=1)
    return pred
