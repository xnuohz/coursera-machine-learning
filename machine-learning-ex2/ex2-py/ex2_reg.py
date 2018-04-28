# -*- coding:utf-8 -*-
import os
import numpy as np
import scipy.optimize as op
from plotData import plotData
from mapFeature import mapFeature
from costFunctionReg import costFunctionReg
from GradientReg import GradientReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict

data = np.loadtxt('ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
f = plotData(X, y)
f.legend(['y = 1', 'y = 0'])
f.show()

# ============ Regularized Logistic Regression ============
X = mapFeature(X[:, 0], X[:, 1])
# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1), dtype='float')
# Set regularization parameter lambda to 1
l = 1
cost = costFunctionReg(initial_theta, X, y, l)
grad = GradientReg(initial_theta, X, y, l)
print('Cost at initial theta: ', cost)
print('Gradient at initial theta: ', grad)
os.system('pause')

# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1], 1), dtype='float')
cost = costFunctionReg(test_theta, X, y, 10)
grad = GradientReg(test_theta, X, y, 10)
print('Cost at test theta: ', cost)
print('Gradient at test theta: ', grad)
os.system('pause')

# ============ Regularzation and Accuracies ============
# Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1), dtype='float')
# Set regularization parameter lambda(0: overfitting, 1: ok, 100: underfitting)
l = 1

res = op.minimize(fun=costFunctionReg,
                  x0=initial_theta,
                  args=(X, y, l),
                  method='TNC',
                  jac=GradientReg)
print('Cost at theta found by fminunc: ', res.fun)
print('theta: ', res.x)
plotDecisionBoundary(res.x, X, y)

p = predict(res.x, X)
print('Train Accuracy: ', round(np.mean(p == y) * 100, 1))
