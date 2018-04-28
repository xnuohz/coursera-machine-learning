# -*- coding:utf-8 -*-
import numpy as np


def emailFeatures(word_indices):
    n = 1899
    features = np.zeros((n + 1,), dtype=int)
    for w in word_indices:
        features[w] = 1
    return features
