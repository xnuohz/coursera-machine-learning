# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor


def plotDataPoints(X, idx, K):
    colors = ['red', 'green', 'blue']
    plt.scatter(X[:, 0], X[:, 1], c=idx,
                cmap=pltcolor.ListedColormap(colors), s=40)
