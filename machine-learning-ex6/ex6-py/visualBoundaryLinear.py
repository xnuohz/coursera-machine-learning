# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from plotData import plotData


def visualBoundaryLinear(X, y, theta, b):
    xb = np.linspace(np.min(X[:, 0]), np.max(X[:, 1]), 100)
    yb = -(theta[0] * xb + b) / theta[1]
    plotData(X, y)
    plt.plot(xb, yb, '-b')
