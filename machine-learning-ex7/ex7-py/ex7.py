# -*- coding:utf-8 -*-
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids

# ============ Find Closest Centroids ============
print('Finding closest centroids.\n')
data2 = scio.loadmat('ex7data2.mat')
X = data2['X']
K = 3
initial_centroids = np.array([3, 3, 6, 2, 8, 5]).reshape(3, 2)
idx = findClosestCentroids(X, initial_centroids)
print('Closest centroids for the first 3 examples: ', idx[:3].flatten())

# ============ Compute Means ============
print('Computing centroids means.\n')
centroids = computeCentroids(X, idx, K)
print(centroids)

# ============ K-Means Clustering ============
print('Running K-Means clustering on example dataset.\n')
max_iters = 10
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print('K-Means Done.\n')

# ============ K-Means Clustering on Pixels ============
print('Running K-Means clustering on pixels from an image.\n')
# A = plt.imread('bird_small.mat)
A = scio.loadmat('bird_small.mat')['A']
A = A / 255
img_size = np.shape(A)
X = A.reshape(img_size[0] * img_size[1], 3)
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
centroids, idx = runkMeans(X, initial_centroids, max_iters)
print(centroids)

# ============ Image Compression ============
idx = findClosestCentroids(X, centroids)
X_recovered = centroids[idx, :]
X_recovered = X_recovered.reshape((img_size[0], img_size[1], 3))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.imshow(A)
ax1.set_title('Original')

ax2 = fig.add_subplot(122)
ax2.imshow(X_recovered)
ax2.set_title('Compressed')

plt.show()
