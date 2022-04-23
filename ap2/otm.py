#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 18:01:34 2022

@author: jose
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

def silh(L):
    return silhouette_score(X = np.dot(L, X.T).T, labels = Y)

def grad(target, L0, delta = 0.001):
    grad = np.zeros(L0.shape)
    for i in range(L0.shape[0]):
        for j in range(L0.shape[1]):
            d = np.zeros(L0.shape)
            d[i, j] = delta
            L = L0 + d
            grad[i, j] = (target(L) - target(L0)) / delta
    return grad

def otm(target, L0, lr = 1, MAX_EPOCAS = 100, delta = 0.001):
    log = []
    for epoca in range(MAX_EPOCAS):
        log.append(target(L0))
        dL = grad(target = target, L0 = L0, delta = delta)
        L0 = L0 - lr * dL
    log.append(target(L0))
    return L0, log

# read data
data = pd.read_csv('data/raw.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]

# otm
# L0 = np.eye(len(X[0, :]))
L0 = np.random.randn(len(X[0, :]), len(X[0, :]))
L, log = otm(silh, L0, lr = -1, MAX_EPOCAS = 100)

# plot
plt.plot(log)
plt.savefig('fig/loss.png')

# write
X = np.dot(L, X.T).T
pd.DataFrame(np.concatenate((X, Y.reshape(-1, 1)), axis = 1)).to_csv('data/trans.csv', header = None, index = None)
