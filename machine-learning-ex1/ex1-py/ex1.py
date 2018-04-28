# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotData import plotData
from computeCost import computeCost
from gradientDescent import gradientDescent

# ============ Plotting ============
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)
y = y.reshape((m, 1))

# Plot Data
plotData(X, y)
# os.system('pause')

# ============ Cost and Gradient descent ============
X = np.concatenate((np.ones((m, 1), dtype='float'), data[:, 0:1]), axis=1)
theta = np.zeros((2, 1), dtype='float')

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('Cost is %f' % J)  # 32.07

J = computeCost(X, y, np.array([(-1), (2)]).reshape((2, 1)))
print('Cost is %f' % J)  # 54.24
os.system('pause')

print('\nRunning Gradient Descent ...\n')
theta, J_history = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:\n')
print(theta, '\n')

# Plot the linear fit
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.legend('Linear regression')
plt.show()

# ============ Visualizing ============
print('Visualizing J(theta_0, theta_1) ...\n')

theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)), dtype='float')

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([(theta0_vals[i]), (theta1_vals[j])]).reshape((2, 1))
        J_vals[i][j] = computeCost(X, y, t)

fig = plt.figure(figsize=(15, 6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

# Left plot
ax1.contour(theta0_vals, theta1_vals, J_vals,
            np.logspace(-2, 3, 20), cmap=plt.cm.jet)
ax1.scatter(theta[0], theta[1], c='r')

# Right plot
ax2.plot_surface(theta0_vals, theta1_vals, J_vals,
                 rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.set_zlim(J_vals.min(), J_vals.max())
ax2.view_init(elev=15, azim=150)

# settings common to both plots
for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

plt.show()
