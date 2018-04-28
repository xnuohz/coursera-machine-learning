# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt


def plotData(x, y):
    '''
    PLOTDATA Plots the data points x and y into a new figure 
    PLOTDATA(x,y) plots the data points and gives the figure axes labels of
    population and profit.
    '''
    plt.plot(x, y, 'rx', LineWidth=1)
    plt.ylabel('Profit in $10,000s')
    plt.xlabel('Population of City in 10,000s')
