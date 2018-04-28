# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from gradientDescentMulti import gradientDescentMulti
from normalEqn import normalEqn

# ============ Feature Normalization ============
print('Loading data ...\n')
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

print('Normalizing Features ...\n')
X, mu, sigma = featureNormalize(X)

# Add intercept term to X
X = np.concatenate([np.ones((m, 1), dtype='float'), X], axis=1)

# ============ Gradient Descent ============
print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1), dtype='float')
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.plot(range(num_iters), J_history, '-b', LineWidth=2)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()

# ============ Normal Equations ============
print('Solving with mormal equations...\n')

data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = len(y)

X = np.concatenate([np.ones((m, 1), dtype='float'), X], axis=1)
theta = normalEqn(X, y)
print(theta)
