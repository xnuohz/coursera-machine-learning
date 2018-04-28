# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from plotDataPoints import plotDataPoints
from drawLine import drawLine


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    plotDataPoints(X, idx, K)
    plt.plot(centroids[:, 0], centroids[:, 1], 'x',
             MarkerEdgeColor='k', ms=10,
             LineWidth=3)
    for j in range(np.size(centroids, 0)):
        drawLine(centroids[j, :], previous[j, :])
    plt.title('Iteration number %d' % (i + 1))
