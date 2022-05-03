#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 21:44:06 2022

@author: jose
"""

import pandas as pd
import numpy as np
from chip import chip
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# read
data = pd.read_csv('data/glass.data', header = None).values

X = data[:, 1:-1]
y = data[:, -1]

# kfold
k = 10
size = len(data)
index = list(range(size))
np.random.shuffle(index)
step = round(size / k)
kfolds = [index[i:i+step] for i in range(0, size, step)]

log = []
for k, kfold in enumerate(kfolds):
    if k==10:
        continue
    fold = np.ones(size, bool)
    fold[kfold] = False
    
    X_test = X[np.invert(fold), :]
    X_train = X[fold, :]
    y_test = y[np.invert(fold)]
    y_train = y[fold]
    
    # norm
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    
    # chip
    chipclass = chip()
    chipclass.fit(X_train, y_train)
    y_hat = chipclass.predict(scaler.transform(X_test))
    log.append(accuracy_score(y_test, y_hat))
print(np.mean(log), np.std(log))
