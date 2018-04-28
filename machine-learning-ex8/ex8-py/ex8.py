# -*- coding:utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from estimateGaussian import estimateGaussian
from multivariateGaussian import multivariateGaussian
from visualizeFit import visualizeFit
from selectThreshold import selectThreshold

# ============ Load Example Dataset ============
print('Visualizing example dataset for outlier detection.\n')
data1 = scio.loadmat('ex8data1.mat')
X, Xval, yval = data1['X'], data1['Xval'], data1['yval'][:, 0]
plt.plot(X[:, 0], X[:, 1], 'bx')
plt.axis([0, 30, 0, 30])
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')
plt.show()

# ============ Estimate the dataset statistics ============
print('Visualizing Gaussian fit.\n')
mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
visualizeFit(X, mu, sigma2)
plt.xlabel('Latency (ms)')
plt.ylabel('Throughput (mb/s)')

# ============ Find Outliers ============
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: ', epsilon)
print('Best F1 on Cross Validation Set: ', F1)

outliers = np.where(p < epsilon)
plt.plot(X[outliers, 0], X[outliers, 1], 'ro', mfc='none', ms=8)
plt.show()

# ============ Multidimensional Outliers ============
data2 = scio.loadmat('ex8data2.mat')
X, Xval, yval = data2['X'], data2['Xval'], data2['yval'][:, 0]
mu, sigma2 = estimateGaussian(X)
p = multivariateGaussian(X, mu, sigma2)
pval = multivariateGaussian(Xval, mu, sigma2)
epsilon, F1 = selectThreshold(yval, pval)
print('Best epsilon found using cross-validation: ', epsilon)
print('Best F1 on Cross Validation Set: ', F1)
print('# Outliers found: ', np.sum(p < epsilon))
