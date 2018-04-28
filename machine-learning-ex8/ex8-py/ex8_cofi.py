# -*- coding:utf-8 -*-
import scipy.io as scio
import scipy.optimize as scop
import numpy as np
import matplotlib.pyplot as plt
from cofiCostFunc import cofiCostFunc
from cofiGradientFunc import cofiGradientFunc

# ============ Loading movie ratings dataset ============
print('Loading movie ratings dataset.\n')
movies = scio.loadmat('ex8_movies.mat')
# y, r: 1682x943
y, r = movies['Y'], movies['R']
print('Average rating for movie 1 (Toy Story): %f / 5\n' %
      np.mean(y[0, r[0, :]]))
plt.imshow(y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()

# ============ Collaborative Filtering Cost Function ============
params = scio.loadmat('ex8_movieParams.mat')
X = params['X']  # 1682x10
Theta = params['Theta']  # 943x10
num_users = params['num_users']  # 1x1
num_movies = params['num_movies']  # 1x1
num_features = params['num_features']  # 1x1
# Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
y = y[0:num_movies, 0:num_users]
r = r[0:num_movies, 0:num_users]
# Evaluate cost function
J = cofiCostFunc(X, Theta, y, r, num_users, num_movies, num_features, 0)
Grad = cofiGradientFunc(X, Theta, y, r, num_users, num_movies, num_features, 0)
print('Cost at loaded parameters: %f(this value should be about 22.22)\n' % J)

# ============ Collaborative Filtering Cost Regularization ============
J = cofiCostFunc(X, Theta, y, r, num_users, num_movies, num_features, 1.5)
Grad = cofiGradientFunc(X, Theta, y, r, num_users,
                        num_movies, num_features, 1.5)
print('Cost at loaded parameters: %f(this value should be about 31.34)\n' % J)
