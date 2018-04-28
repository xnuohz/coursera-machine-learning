# -*- coding:utf-8 -*-
import numpy as np


def mapFeature(x1, x2):
    degree = 6
    out = np.ones((x1.shape[0], 1))
    # i从1开始，从0开始又是另外一幅图，调了半天
    for i in range(1, degree + 1):
        for j in range(i + 1):
            newColumn = np.multiply(x1 ** (i - j), x2 ** j)
            out = np.column_stack((out, newColumn))
    return out
