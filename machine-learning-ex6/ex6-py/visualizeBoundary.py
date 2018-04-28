# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData


def visualBoundary(X, y, model):
    x1plot = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    x2plot = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(np.shape(X1), dtype=float)
    plotData(X, y)
    for i in range(np.size(X1, 1)):
        this_X = np.vstack((X1[:, i], X2[:, i])).T
        vals[:, i] = model.predict(this_X)
    plt.contour(X1, X2, vals, colors='b')
