# -*- coding:utf-8 -*-
"""
DISPLAYDATA Display 2D data in a nice grid
[h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
stored in X in a nice grid. It returns the figure handle h and the 
displayed array if requested.
"""
from math import *
import numpy as np
import matplotlib.pyplot as plt


def displayData(X):
    m, n = np.shape(X)
    example_width = round(sqrt(n))
    # 去掉int就gg
    example_height = int(n/ example_width)

    # Compute number of items to display
    display_rows = floor(sqrt(m))
    display_cols = ceil(m / display_rows)

    # Between images padding
    pad = 1

    # Setup blank display
    a = pad + display_rows * (example_height + pad)
    b = pad + display_cols * (example_width + pad)
    display_array = -np.ones((a, b), dtype=float)

    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(X[curr_ex, :]))
            i_1 = pad + j * (example_height + pad)
            i_2 = pad + i * (example_width + pad)
            display_array[i_1:i_1 + example_height, i_2:i_2 + example_width] = X[curr_ex, :].reshape(example_height, example_width) / max_val
            curr_ex += 1
        if curr_ex >= m:
                break
    plt.imshow(display_array.T, cmap='gray')
    plt.show()