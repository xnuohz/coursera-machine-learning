# -*- coding:utf-8 -*-
from math import *
import numpy as np
import matplotlib.pyplot as plt


def displayData(X):
    # Compute rows, cols 100x400
    m, n = np.shape(X)
    example_width = round(sqrt(n))
    example_height = int(n / example_width)
    # Compute number of items to display
    display_rows = floor(sqrt(m))
    display_cols = ceil(m / display_rows)
    # Between images padding
    pad = 1
    # Setup blank display
    row = pad + display_rows * (example_height + pad)
    col = pad + display_cols * (example_width + pad)
    display_array = -np.ones((row, col), dtype=float)
    # Copy each example into a patch on the display array
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch
            # Get the max value of the patch
            max_val = max(abs(X[curr_ex, :]))
            i_1 = pad + j * (example_height + pad)
            i_2 = pad + i * (example_width + pad)
            display_array[i_1:i_1 + example_height, i_2:i_2 + example_width] = X[curr_ex,
                                                                                 :].reshape(example_height, example_width).T / max_val
            curr_ex += 1
        if curr_ex >= m:
            break
    plt.imshow(display_array, cmap='gray')
