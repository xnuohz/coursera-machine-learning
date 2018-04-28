# -*- coding:utf-8 -*-
'''
PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
the decision boundary defined by theta
PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
positive examples and o for the negative examples. X is assumed to be 
a either 
1) Mx3 matrix, where the first column is an all-ones column for the 
intercept.
2) MxN, N>3 matrix, where the first column is all-ones
'''
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from mapFeature import mapFeature


def plotDecisionBoundary(theta, X, y):
    f = plotData(X[:, 1:], y)
    m, n = X.shape
    if n <= 3:
        plot_x = np.array([min(X[:, 1]) - 2, max(X[:, 1]) + 2])
        plot_y = -(plot_x.dot(theta[1]) + theta[0]) / theta[2]
        f.plot(plot_x, plot_y, label='Test Data', color='b')
        plt.show()
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((len(u), len(v)), dtype='float')
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = mapFeature(
                    np.array([u[i]]), np.array([v[j]])).dot(theta)
        z = z.T
        plt.contour(u, v, z)
        plt.show()
