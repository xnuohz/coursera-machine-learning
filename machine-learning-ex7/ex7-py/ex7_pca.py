# -*- coding:utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as pltcolor
from featureNormalize import featureNormalize
from pca import pca
from drawLine import drawLine
from projectData import projectData, recoverData
from displayData import displayData
from kMeansInitCentroids import kMeansInitCentroids
from runkMeans import runkMeans
from plotDataPoints import plotDataPoints

# ============ Load Example Dataste ============
print('Visualizing example dataset for PCA.\n')
data1 = scio.loadmat('ex7data1.mat')
X = data1['X']
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])

# ============ Principal Component Analysis =============
print('Running PCA on example dataset.\n')
X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)
drawLine(mu, mu + 1.5 * S[0] * U[:, 0])
drawLine(mu, mu + 1.5 * S[1] * U[:, 1])
plt.show()

# ============ Dimension Reduction ============
print('Dimension reduction on example dataset.\n')
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4, 3, -4, 3])
plt.show()

K = 1
Z = projectData(X_norm, U, K)
print(Z.flatten())
X_rec = recoverData(Z, U, K)
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
for i in range(np.size(X_norm, 0)):
    drawLine(X_norm[i, :], X_rec[i, :])
plt.show()

# ============ Loading and Visualizing Face Data ============
print('Loading face dataset.\n')
face = scio.loadmat('ex7faces.mat')
X = face['X']  # 5000x1024
displayData(X[:100, :])
plt.show()

# ============ PCA on Face Data: Eigenfaces ============
print('Running PCA on face dataset.\n(this might take a minute or two ...)\n')
X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)
displayData(U[:, 0:36].T)
plt.show()

# ============ Dimension Reduction for Faces ============
print('Dimension reduction for face dataset.\n')
K = 100
Z = projectData(X_norm, U, K)
print('The projected data Z has a size of: ', np.size(Z))

# ============ Visualizing of Faces after PCA Dimension Reduction ============
print('Visualizing the projected (reduced dimension) faces.\n')
K = 100
X_rec = recoverData(Z, U, K)
plt.subplot(121)
displayData(X_norm[0:100, :])
plt.title('Original faces')
plt.subplot(122)
displayData(X_rec[0:100, :])
plt.title('Recovered faces')
plt.show()

# ============ (a) Optional (ungraded) Exercise: PCA for Visualization ============
A = plt.imread('bird_small.png')
A = A / 255
img_size = np.shape(A)
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)
sel = np.floor(np.random.rand(1000,) * np.size(X, 0)).astype(int) + 1
colors = cm.rainbow(np.linspace(0, 1, K))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], c=idx[sel],
           cmap=pltcolor.ListedColormap(colors), marker='o')
ax.set_title('Pixel dataset plotted in 3D. Color shows centroid memberships')
plt.show()

# ============ (b) Optional (ungraded) Exercise: PCA for Visualization ============
X_norm, mu, sigma = featureNormalize(X)
U, S = pca(X_norm)
Z = projectData(X_norm, U, 2)
colors = cm.rainbow(np.linspace(0, 1, K))
plt.scatter(Z[sel, 0], Z[sel, 1], c=idx[sel], cmap=pltcolor.ListedColormap(colors), marker='o')
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()

