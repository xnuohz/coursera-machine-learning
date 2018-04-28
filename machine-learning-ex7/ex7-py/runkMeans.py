# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from plotProgresskMeans import plotProgresskMeans


def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    m, n = np.shape(X)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1), dtype=float)

    if plot_progress:
        plt.ion()
        fig = plt.figure()

    for i in range(max_iters):
        print('K-Means iteration %d/%d...\n' % (i + 1, max_iters))
        idx = findClosestCentroids(X, centroids)
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            fig.canvas.draw()
            os.system('pause')
        centroids = computeCentroids(X, idx, K)
    plt.show(block=True)
    plt.ioff()
    return centroids, idx
