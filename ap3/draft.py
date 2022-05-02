#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:47:17 2022

@author: jose
"""

import pandas as pd
import numpy as np
# from gg import lara_graph, overlap, suporte, pares, pesos, classificador, f
from chip import chip
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# read
data = pd.read_csv('data/spirals.csv', header = None).values
# data = pd.read_csv('data/overlap.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]
Y[Y == 2] = -1

# train/test
size = len(data)
index = list(range(size))
np.random.shuffle(index)

train = index[0:int(0.7*size)]
fold = np.zeros(size, bool)
fold[train] = True

X_test = X[np.invert(fold), :]
X_train = X[fold, :]
Y_test = Y[np.invert(fold)]
Y_train = Y[fold]

# chip
chipclass = chip()
chipclass.fit(X_train, Y_train, True)
y_hat = chipclass.predict(X_test, 0.1)
print(accuracy_score(Y_test, y_hat))

# plot
for classe in np.unique(Y):
    idx = Y == classe
    _idx = Y_test == classe
    plt.scatter(X[idx, 0], X[idx, 1], c = {1: 'red', -1: 'blue'}[classe])
    plt.scatter(X_test[_idx, 0], X_test[_idx, 1], c = {1: 'red', -1: 'blue'}[classe])
# plt.scatter(X[fronteira, 0], X[fronteira, 1], c = 'green')

# contorno
N = 50
xx = np.linspace(-1, 1, N)
yy = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(xx, yy)
Z = np.zeros(shape = XX.shape)
for i in range(N):
    for j in range(N):
        x = XX[i, j]
        y = YY[i, j]
        Z[i, j] = chipclass.predict(np.array([x, y]).reshape(1, 2), 0.1)
plt.contour(XX, YY, Z, [-1, 1])
