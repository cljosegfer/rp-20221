#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:47:17 2022

@author: jose
"""

import pandas as pd
import numpy as np
from gg import lara_graph, overlap, suporte, pares, pesos, classificador, f
from chip import chip
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# read
data = pd.read_csv('data/spirals.csv', header = None).values
# data = pd.read_csv('data/overlap.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]
Y[Y == 2] = -1

# # train/test
# size = len(data)
# index = list(range(size))
# np.random.shuffle(index)

# train = index[0:int(0.7*size)]
# fold = np.zeros(size, bool)
# fold[train] = True

# kfold
k = 10
size = len(data)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

log = []
for k, kfold in enumerate(kfolds):
    print(k)
    fold = np.ones(size, bool)
    fold[kfold] = False
    
    X_test = X[np.invert(fold), :]
    X_train = X[fold, :]
    Y_test = Y[np.invert(fold)]
    Y_train = Y[fold]
    
    # chip
    chipclass = chip()
    chipclass.fit(X_train, Y_train, True)
    y_hat = chipclass.predict(X_test)
    log.append(accuracy_score(Y_test, y_hat))
print(np.mean(log), np.std(log))

adjacencia = lara_graph(X)
X, Y, adjacencia, _ = overlap(X, Y, adjacencia)
fronteira = suporte(X, Y, adjacencia)

# plot
for classe in np.unique(Y):
    idx = Y == classe
    _idx = Y_test == classe
    plt.scatter(X[idx, 0], X[idx, 1], c = {1: 'red', -1: 'blue'}[classe])
    plt.scatter(X_test[_idx, 0], X_test[_idx, 1], c = {1: 'red', -1: 'blue'}[classe])
plt.scatter(X[fronteira, 0], X[fronteira, 1], c = 'green')
plt.savefig('fig/data.png')

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
        Z[i, j] = chipclass.predict(np.array([x, y]).reshape(1, 2))
plt.contour(XX, YY, Z, [-1, 1])
plt.savefig('fig/contour.png')
