# -*- coding:utf-8 -*-
import numpy as np
import scipy.linalg as slin
from debugInitializeWeights import debugInitializeWeights
from nnCostFunction import nnCostFunction
from nnGradient import nnGradient
from computeNumericalGradient import computeNumericalGradient


def checkNNGradients(l):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    x = debugInitializeWeights(m, input_layer_size - 1)
    y = 1 + (np.arange(m) + 1) % num_labels

    nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))

    cost = nnCostFunction(nn_params, input_layer_size,
                          hidden_layer_size, num_labels, x, y, l)
    grad = nnGradient(nn_params, input_layer_size,
                      hidden_layer_size, num_labels, x, y, l)
    numgrad = computeNumericalGradient(nnCostFunction, nn_params,
                                       (input_layer_size, hidden_layer_size, num_labels, x, y, l))
    print(numgrad, '\n', grad)
    print('The above two columns you get should be very similar.\n \
    (Left-Your Numerical Gradient, Right-Analytical Gradient)')
    diff = slin.norm(numgrad - grad) / slin.norm(numgrad + grad)
    print('If your backpropagation implementation is correct, then \n\
         the relative difference will be small (less than 1e-9). \n\
         \nRelative Difference: ', diff)
