# -*- coding:utf-8 -*-
import numpy as np


def debugInitializeWeights(f_out, f_in):
    w = np.sin(np.arange(f_out * (f_in + 1)) + 1).reshape(f_out, f_in + 1) / 10
    return w
