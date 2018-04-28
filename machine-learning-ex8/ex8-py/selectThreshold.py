# -*- coding:utf-8 -*-
import numpy as np


def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000
    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # 真正例
        tp = sum((yval == 1) & (pval < epsilon))
        # 假正例
        fp = sum((yval == 1) & (pval >= epsilon))
        # 假反例
        fn = sum((yval == 0) & (pval < epsilon))
        if tp + fp == 0 or tp + fn == 0:
            F1 = -1
        else:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            F1 = 2 * prec * rec / (prec + rec)

        if F1 > bestF1:
            bestEpsilon = epsilon
            bestF1 = F1
    return bestEpsilon, bestF1
