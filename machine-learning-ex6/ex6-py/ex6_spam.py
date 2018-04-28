# -*- coding:utf-8 -*-
import numpy as np
import scipy.io as scio
from sklearn import svm
from readFile import readFile
from processEmail import processEmail
from emailFeatures import emailFeatures
from getVocabList import getVocabListReverse

# ============ Email Preprocessing ============
print('Preprocessing sample email (emailSample1.txt)\n')
file_contents = readFile('emailSample1.txt')
word_indices = processEmail(file_contents)
print('Word Indices:', word_indices)

# ============ Feature Extraction ============
print('Extracting features from sample email (emailSample1.txt)\n')
features = emailFeatures(word_indices)
print('Length of feature vector: ', features.size)
print('Number of non-zero entries: ', np.flatnonzero(features).size)

# ============= Train Linear SVM for Spam Classification ============
spam_train = scio.loadmat('spamTrain.mat')
X, y = spam_train['X'], spam_train['y'][:, 0]

print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes)\n')

model = svm.SVC(C=0.1, kernel='linear')
model.fit(X, y)
p = model.predict(X)

print('Training Accuracy: ', np.mean(p == y) * 100)

# ============ Test Spam Classification ============
print('Evaluating the trained Linear SVM on a test set ...\n')
spam_test = scio.loadmat('spamTest.mat')
Xtest, ytest = spam_test['Xtest'], spam_test['ytest'][:, 0]

p = model.predict(Xtest)

print('Test Accuracy: ', np.mean(p == ytest) * 100)

# ============ Top Predictors of Spam ============
vocab = getVocabListReverse()
indices = np.argsort(model.coef_).flatten()[::-1]
for i in range(15):
    print(vocab[indices[i]], model.coef_.flatten()[indices[i]])
