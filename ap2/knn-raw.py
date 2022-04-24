#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:54:15 2022

@author: jose
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# read
data = pd.read_csv('data/raw.csv', header = None).values

# kfold
k = 10
size = len(data)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

for n in range(15):
    n += 1
    log = []

    for k, kfold in enumerate(kfolds):
        fold = np.ones(size, bool)
        fold[kfold] = False
        # train
        X_train = data[fold, 0:-1]
        Y_train = data[fold, -1]
        # test
        X_test = data[np.invert(fold), 0:-1]
        Y_test = data[np.invert(fold), -1]
    
        # knn
        model = KNeighborsClassifier(n_neighbors = n)
        model.fit(X_train, Y_train)
        
        # eval
        acc = model.score(X_test, Y_test)
        log.append(acc)
    
    print(n, np.mean(log), np.std(log))
    