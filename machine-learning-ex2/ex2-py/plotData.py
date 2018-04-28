# -*- coding:utf-8 -*-
'''
PLOTDATA Plots the data points X and y into a new figure 
PLOTDATA(x,y) plots the data points with + for the positive examples
and o for the negative examples. X is assumed to be a Mx2 matrix.
'''
import matplotlib.pyplot as plt
import numpy as np


def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos, 0], X[pos, 1], 'k+',
             LineWidth=2, MarkerSize=7)
    plt.plot(X[neg, 0], X[neg, 1], 'ko',
             MarkerFaceCOlor='y', MarkerSize=7)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(['Admitted', 'Not admitted'])
    return plt
