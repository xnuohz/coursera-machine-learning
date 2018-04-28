# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import os
from linearRegCostFunction import linearRegCostFunction
from linearRegGradient import linearRegGradient
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from polyFeatures import polyFeatures
from featureNormalize import featureNormalize
from plotFit import plotFit
from validationCurve import validationCurve

# ============ Loading and Visualizing Data ===========
print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex5data1.mat')
X, y = data['X'], data['y'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]

m = np.size(X, 0)
mval = np.size(Xval, 0)
mtest = np.size(Xtest, 0)
plt.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
os.system('pause')

t = np.concatenate((np.ones((m, 1), dtype=float), X), axis=1)
tval = np.concatenate((np.ones((mval, 1), dtype=float), Xval), axis=1)

# ============ Regularized Linear Regression Cost ============
theta = np.array([1, 1])
J = linearRegCostFunction(theta, t, y, 1)
print('Cost: ', J)
os.system('pause')

# ============ Regularized Linear Regression Gradient ============
grad = linearRegGradient(theta, t, y, 1)
print('Gradient: ', grad)
os.system('pause')

# ============ Train Linear Regression ============
# Train linear regression with lambda = 0
l = 0
# 貌似fmin_cg更适合(x,)类型的theta，狗屎
theta = trainLinearReg(t, y, l)

# Plot fit over the data
plt.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, t.dot(theta), '--', LineWidth=2)
plt.show()
os.system('pause')

# ============ Learning Curve for Linear Regression ============
l = 0
error_train, error_val = learningCurve(t, y, tval, yval, l)
plt.plot(range(m), error_train, range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.show()
os.system('pause')

# ============ Feature Mapping for Polynomial Regression ============
p = 8
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)
X_poly = np.concatenate((np.ones((m, 1), dtype=float), X_poly), axis=1)

X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.concatenate(
    (np.ones((mtest, 1), dtype=float), X_poly_test), axis=1)

X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.concatenate(
    (np.ones((mval, 1), dtype=float), X_poly_val), axis=1)

# ============ Learning Curve for Polynomial Regression ============
l = 0
theta = trainLinearReg(X_poly, y, l)
X_simu, y_simu = plotFit(np.min(X), np.max(X), mu, sigma, theta, p)

# Plot training data and fit
f1 = plt.figure(1)
ax1 = f1.add_subplot(111)
ax1.plot(X, y, 'rx', MarkerSize=10, LineWidth=1.5)
ax1.plot(X_simu, y_simu, '--', LineWidth=2)
ax1.set_xlabel('Change in water level (x)')
ax1.set_ylabel('Water flowing out of the dam (y)')
f1.suptitle('Polynomial Regression Fit (lambda = 0)')

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, l)
f2 = plt.figure(2)
ax2 = f2.add_subplot(111)
ax2.plot(range(m), error_train, range(m), error_val)
ax2.set_xlabel('Number of training examples')
ax2.set_ylabel('Error')
ax2.set_xlim(0, 13)
ax2.set_ylim(0, 100)
ax2.legend(['Train', 'Cross Validation'])
f2.suptitle('Polynomial Regression Learning Curve (lambda = 0)')
plt.show()

print('Polynomial Regression (lambda = 0)')
print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))

# ============ Validation for Selecting Lambda ==============
lambda_vec, err_train, err_val = validationCurve(X_poly, y, X_poly_val, yval)
plt.plot(lambda_vec, err_train, 'b', label='Train')
plt.plot(lambda_vec, err_val, 'r', label='Cross Validation')
plt.xlabel('lambda')
plt.ylabel('Error')
plt.legend(loc='upper right')
plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[i], err_train[i], err_val[i]))
