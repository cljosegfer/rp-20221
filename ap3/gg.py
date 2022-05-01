#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:47 2022

@author: jose
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix

def gg(X):
    D = distance_matrix(X, X) ** 2
    D[np.diag_indices(D.shape[0])] = 1e6

    n = X.shape[0]
    adjacencia = np.zeros(shape = (n, n))
    for i in range(n-1):
        for j in range(i+1, n):
            minimo = min(D[i, :] + D[j, :])
            if (D[i, j] <= minimo):
                adjacencia[i, j] = 1
                adjacencia[j, i] = 1
    return adjacencia

# read data
data = pd.read_csv('data/raw.csv', header = None).values
gg = pd.read_csv('data/raw-gg.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]

# gg
D = distance_matrix(X, X) ** 2
D[np.diag_indices(D.shape[0])] = 1e6

n = X.shape[0]
adjacencia = np.zeros(shape = (n, n))
for i in range(n-1):
    for j in range(i+1, n):
        # print(i, j)
        minimo = min(D[i, :] + D[j, :])
        if (D[i, j] <= minimo):
            adjacencia[i, j] = 1
            adjacencia[j, i] = 1
