# -*- coding:utf-8 -*-
import os
import scipy.io as scio
import numpy as np
from displayData import displayData
from predict import predict

# Setup the parameters you will use for this exercise
# 20x20 Input Images of Digits
input_layer_size = 400
# 25 hidden units
hidden_layer_size = 25
# 10 labels, from 1 to 10, note that 0 => 10
num_labels = 10

# ============ Loading and Visualizing Data ============
print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex3data1.mat')
X, y = data['X'], data['y'][:, 0] % 10
m = np.shape(X)[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]
displayData(X[sel, :])
os.system('pause')

# ============ Loading Parameters ============
print('Loading Saved Neural Network Parameters ...\n')
params = scio.loadmat('ex3weights.mat')
theta1, theta2 = params['Theta1'], params['Theta2']  # 25x401, 10x26

# ============ Implement Predict ============
pred = predict(theta1, theta2, X)
print('Traning Set Accuracy: ', np.mean(pred == y) * 100)
os.system('pause')

rp = np.random.permutation(m)

for i in range(m):
    print('Displaying Example Image\n')
    t = X[rp[i], :].reshape((1, 400))
    displayData(t)

    pred = predict(theta1, theta2, t)
    print('Neural Network Prediction: %d (digit %d)' % (pred, pred % 10))

    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break
