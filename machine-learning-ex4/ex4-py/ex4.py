# -*- coding:utf-8 -*-
import os
import scipy.io as scio
import scipy.optimize as scop
import numpy as np
from displayData import displayData
from sigmoidGradient import sigmoidGradient
from nnCostFunction import nnCostFunction
from nnGradient import nnGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict

# 20x20 Input Images of Digits
input_layer_size = 400
# 25 hidden units
hidden_layer_size = 25
# 10 labels, from 1 to 10, note that 0 => 10
num_labels = 10

# ============ Loading and Visualizing Data ============
# Load Training Data
print('Loading and Visualizing Data ...\n')
data = scio.loadmat('ex4data1.mat')
X = data['X']
y = data['y'][:, 0]
m = np.shape(X)[0]

# Randomly select 100 data points to display
sel = np.random.permutation(m)
sel = sel[0:100]

# displayData(X[sel, :])
# os.system('pause')

# ============ Loading Parameters ============
print('Loading Saved Neural Network Parameters ...\n')
# Load the weights into variables Theta1 and Theta2
params = scio.loadmat('ex4weights.mat')
theta1 = params['Theta1']  # 25x401
theta2 = params['Theta2']  # 10x26
nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))

# ============ Compute Cost (Feedforward) ============
print('Feedforward Using Neural Network ...\n')

# Weight regularization parameter
l = 0

J = nnCostFunction(nn_params, input_layer_size,
                   hidden_layer_size, num_labels,
                   X, y, l)
grad = nnGradient(nn_params, input_layer_size,
                  hidden_layer_size, num_labels,
                  X, y, l)
print('Cost: ', J)
os.system('pause')

# ============ Implement Regularization ============
print('Checking Cost Function (w/ Regularization) ...\n')
l = 1

J = nnCostFunction(nn_params, input_layer_size,
                   hidden_layer_size, num_labels, X, y, l)
grad = nnGradient(nn_params, input_layer_size,
                  hidden_layer_size, num_labels, X, y, l)
print('Cost: ', J)
os.system('pause')

# ============ Sigmoid Gradient ============
print('Evaluating sigmoid gradient ...\n')
g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient: ', g)
os.system('pause')

# ============ Initializing Parameters ============
print('Initializing Neural Network Parameters ...\n')

initial_theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.concatenate(
    (initial_theta1.flatten(), initial_theta2.flatten()))

# ============ Implement Backpropgation ============
print('Checking Backpropgation ...\n')
checkNNGradients(0)
os.system('pause')

# ============ Implement Regularization ============
print('Checking Backpropgation (w/ Regularization) ...\n')
l = 3
checkNNGradients(l)

# Also output the costFunction debugging values
debug_J = nnCostFunction(nn_params, input_layer_size,
                         hidden_layer_size, num_labels, X, y, l)
print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f \n(this value should be about 0.576051)' %
      debug_J)
os.system('pause')

# ============ Training NN ============
print('Training Neural Network... \n')

l = 1
param = scop.fmin_cg(nnCostFunction, initial_nn_params,
                     fprime=nnGradient,
                     args=(input_layer_size, hidden_layer_size,
                           num_labels, X, y, l),
                     maxiter=50)
theta1 = param[0:hidden_layer_size * (input_layer_size + 1)]
theta1 = theta1.reshape(hidden_layer_size, input_layer_size + 1)
theta2 = param[hidden_layer_size * (input_layer_size + 1):]
theta2 = theta2.reshape(num_labels, hidden_layer_size + 1)
os.system('pause')

# ============ Visualize Weights ============
print('Visualizing Neural Network ...')
displayData(theta1[:, 1:])
os.system('pause')

# ============ Implement Predict ============
pred = predict(theta1, theta2, X)
print('Training Set Accuracy: ', np.mean(pred == y) * 100)
