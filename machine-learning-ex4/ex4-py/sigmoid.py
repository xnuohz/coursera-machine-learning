# -*- coding:utf-8 -*-
'''
SIGMOID Compute sigmoid function
g = SIGMOID(z) computes the sigmoid of z.
'''
import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
