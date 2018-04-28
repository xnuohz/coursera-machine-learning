# -*- coding:utf-8 -*-
"""
NNCOSTFUNCTION Implements the neural network cost function for a two layer
neural network which performs classification
[J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
X, y, lambda) computes the cost and gradient of the neural network. The
parameters for the neural network are "unrolled" into the vector
nn_params and need to be converted back into the weight matrices. 

The returned parameter grad should be a "unrolled" vector of the
partial derivatives of the neural network.
"""
import numpy as np
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient


def nnCostFunction(nn_params, input_layer_size,
                   hidden_layer_size, num_labels,
                   X, y, l):
    """
    theta1 25x401
    theta2 10x26
    """
    theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)]
    theta1 = theta1.reshape((hidden_layer_size, input_layer_size + 1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):]
    theta2 = theta2.reshape((num_labels, hidden_layer_size + 1))
    m = np.shape(X)[0]

    a1 = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1)
    z2 = a1.dot(theta1.T)  # 5000x25
    a2 = np.concatenate((np.ones((m, 1)), sigmoid(z2)), axis=1)  # 5000x26
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)  # 5000x10
    p = np.zeros((m, num_labels), dtype=int)
    p[np.arange(m), y - 1] = 1
    j = np.sum(-p * np.log(a3) - (1 - p) * np.log(1 - a3)) / m

    reg = np.sum(
        np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))
    j += l / (2 * m) * reg
    return j
