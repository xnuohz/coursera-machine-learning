# -*- coding:utf-8 -*-
import numpy as np


def randInitializeWeights(l_in, l_out):
    epsilon_init = 0.12
    w = np.random.rand(l_out, l_in + 1) * 2 * epsilon_init - epsilon_init
    return w
