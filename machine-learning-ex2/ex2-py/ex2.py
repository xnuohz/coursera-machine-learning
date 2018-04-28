# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.optimize as op
from plotData import plotData
from costFunction import costFunction
from Gradient import Gradient
from plotDecisionBoundary import plotDecisionBoundary
from sigmoid import sigmoid
from predict import predict

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# ============ Plotting ============
print('Plotting data ...\n')
f = plotData(X, y)
f.show()
os.system('pause')

# ============ Compute Cost and Gradient ============
m, n = np.shape(X)
# Add intercept term to x and X_test
X = np.concatenate([np.ones((m, 1), dtype='float'),
                    X], axis=1)
# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1), dtype='float')
# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
grad = Gradient(initial_theta, X, y)
print('Cost: %f' % cost)
print('Gradient: ', grad)

# Compute and display cost gradient with non-zero theta
test_theta = np.array([-24, 0.2, 0.2]).reshape(3, 1)
cost = costFunction(test_theta, X, y)
grad = Gradient(test_theta, X, y)
print('Cost at test theta: %f' % cost)
print('Gradient at test theta: ', grad)
os.system('pause')

# ============ Optimizing using fminunc ============
res = op.minimize(fun=costFunction,
                  x0=initial_theta,
                  args=(X, y),
                  method='TNC',
                  jac=Gradient)
print('Cost at theta found by fminunc: ', res.fun)
print('theta: ', res.x)

# Plot Boundary
plotDecisionBoundary(res.x, X, y)
os.system('pause')

# ============ Predict and Accuracies ============
prob = sigmoid(np.array([1, 45, 85]).dot(res.x))
print('For a student with scores 45 and 85, we predict an admission', prob)

p = predict(res.x, X)
print('Train Accuracy: ', np.mean(p == y) * 100)
