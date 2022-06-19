#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:22:54 2022

@author: jose
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# read
data = pd.read_csv('data/raw.csv', header = None).values

# kfold
k = 10
size = len(data)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

k = 0
kfold = kfolds[0]
fold = np.ones(size, bool)
fold[kfold] = False

# data
X_train = data[fold, 0:-1]
y_train = data[fold, -1]
X_test = data[np.invert(fold), 0:-1]
y_test = data[np.invert(fold), -1]

# svm
C = 1
h = 1
# clf = make_pipeline(StandardScaler(), SVC(C = C, gamma = gamma))
clf = SVC(C = C, gamma = h)
clf.fit(X_train, y_train)
# print(clf.get_params()['gamma'])

# eval
yhat = clf.predict(X_test)
print(clf.score(X_test, y_test))

# plot
X = data[:, 0:-1]
y = data[:, -1]
for classe in np.unique(y):
    idx = y_train == classe
    _idx = y_test == classe
    plt.scatter(X_train[idx, 0], X_train[idx, 1], c = {1: 'red', -1: 'blue'}[classe])
    plt.scatter(X_test[_idx, 0], X_test[_idx, 1], c = {1: 'red', -1: 'blue'}[classe])
fronteira = clf.support_
# plt.scatter(X[fronteira, 0], X[fronteira, 1], c = 'green')

# contorno
N = 100
li = 0
ls = 6
xx = np.linspace(li, ls, N)
yy = np.linspace(li, ls, N)
XX, YY = np.meshgrid(xx, yy)
Z = np.zeros(shape = XX.shape)
for i in range(N):
    for j in range(N):
        x = XX[i, j]
        y = YY[i, j]
        Z[i, j] = clf.predict(np.array([x, y]).reshape(1, 2))
plt.contour(XX, YY, Z)
# plt.imshow(Z)
