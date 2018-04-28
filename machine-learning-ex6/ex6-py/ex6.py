# -*- coding:utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from sklearn import svm
from visualBoundaryLinear import visualBoundaryLinear
from gaussianKernel import gaussianKernel
from visualizeBoundary import visualBoundary
from dataset3Params import dataset3Params

# ============ Loading and Visualizing Data ============
print('Loading and Visualizing Data ...\n')
data1 = scio.loadmat('ex6data1.mat')
X, y = data1['X'], data1['y'][:, 0]
plotData(X, y)
plt.show()

# ============ Training Linear SVM ============
print('Training Linear SVM ...\n')
model = svm.SVC(C=1, kernel='linear')
model.fit(X, y)
theta = model.coef_.flatten()
b = model.intercept_
visualBoundaryLinear(X, y, theta, b)
plt.show()

# ============ Implement Gaussian Kernel ============
print('Evaluating the Gaussian Kernel ...\n')
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)
print('Gaussian Kernel: ', sim)

# ============ Visualizing Dataset 2 ============
print('Loading and Visualizing Data ...\n')
data2 = scio.loadmat('ex6data2.mat')
X, y = data2['X'], data2['y'][:, 0]
plotData(X, y)
plt.show()

# ============ Training SVM with RBF Kernel (Dataset 2) ============
print('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n')
sigma = 0.1
model = svm.SVC(C=1, kernel='rbf', gamma=1 / (2 * sigma**2))
model.fit(X, y)
visualBoundary(X, y, model)
plt.show()

# ============ Visualizing Dataset3 ============
print('Loading and Visualizing Data ...\n')
data3 = scio.loadmat('ex6data3.mat')
X, y = data3['X'], data3['y'][:, 0]
Xval, yval = data3['Xval'], data3['yval'][:, 0]
plotData(X, y)
plt.show()

# ============ Training SVM with RBF Kernel (Dataset 3) ============
c, sigma = dataset3Params(X, y, Xval, yval)
model = svm.SVC(C=c, kernel='rbf', gamma=1 / (2 * sigma**2))
model.fit(X, y)
visualBoundary(X, y, model)
plt.show()
