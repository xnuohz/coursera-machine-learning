# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def drawLine(p1, p2):
    """ 中心点连线 """
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])
    plt.plot(x, y)
