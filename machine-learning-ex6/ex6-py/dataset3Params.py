# -*- coding:utf-8 -*-
import numpy as np
from sklearn import svm


def dataset3Params(X, y, Xval, yval):
    c = sigma = 0
    lambda_all = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]).T
    len_all = np.size(lambda_all, 0)
    errorVal = 999999999

    for i in range(len_all):
        now_c = lambda_all[i]
        for j in range(len_all):
            now_sigma = lambda_all[j]
            model = svm.SVC(C=now_c, kernel='rbf',
                            gamma=1 / (2 * now_sigma**2))
            model.fit(X, y)
            pred = model.predict(Xval)
            error = np.mean(pred != yval)
            if error < errorVal:
                errorVal = error
                c = now_c
                sigma = now_sigma
    return c, sigma
