# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)

    plt.plot(X[pos, 0], X[pos, 1], 'k+', LineWidth=1, MarkerSize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', MarkerFaceColor='y', MarkerSize=7)
