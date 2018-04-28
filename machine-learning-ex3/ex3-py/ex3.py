# -*- coding:utf-8 -*-
import os
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from displayData import displayData
from lrCostFunction import lrCostFunction
from gradient import gradient
from gradient_SB import gradient_SB
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

# 20x20 Input Images of Digits
input_layer_size = 400
# 10 labels, from 1 to 10
num_labels = 10

# ============ Loading and Visualizing Data ============
# Load Training Data
print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex3data1.mat')
X, y = data['X'], data['y'][:, 0] % 10
m = np.shape(X)[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m)
sel = X[rand_indices[0:100], :]
displayData(sel)
os.system('pause')

# ============ Vectorize Logistic Regression ============
print('\nTesting lrCostFunction() with regularization')
theta_t = np.array([-2, -1, 1, 2]).reshape(-1, 1)  # 4x1
t1 = np.ones((5, 1), dtype=float)  # 5x1
t2 = np.reshape(range(1, 16), (3, 5)).T / 10  # 5x3
X_t = np.c_[t1, t2] # 5x4
y_t = np.array([1, 0, 1, 0, 1]).reshape(-1, 1) >= 0.5  # 5x1
lambda_t = 3
J = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = gradient_SB(theta_t, X_t, y_t, lambda_t)

print('Cost: ', J)
print('Gradients: ', grad.flatten())
os.system('pause')

# ============ One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

lam = 0.1
all_theta = oneVsAll(X, y, num_labels, lam)
os.system('pause')

# ============ Predict for One-Vs-All ============
pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy:', np.mean(pred == y) * 100)
