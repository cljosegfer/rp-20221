#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 10:27:09 2022

@author: jose
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# kfold
K = 10
size = 200
step = round(size / K)

log = []

for k in range(K):
    # read
    data = pd.read_csv('data/kfold/trans-' + str(k) + '.csv', header = None).values
    X = data[:, 0:-1]
    Y = data[:, -1]

    # train/test
    X_train = X[step:, :]
    Y_train = Y[step:]
    X_test = X[:step, :]
    Y_test = Y[:step]

    # knn
    model = KNeighborsClassifier(n_neighbors = 3)
    model.fit(X_train, Y_train)

    # log
    acc = model.score(X_test, Y_test)
    log.append(acc)

print(np.mean(log), np.std(log))
