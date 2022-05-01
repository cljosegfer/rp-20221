#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:36:47 2022

@author: jose
"""

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

def lara_graph(X):
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

def suporte(X, Y, adjacencia):
    fronteira = []
    u, _ = np.unique(Y, return_index = True)
    for c in u[np.argsort(_)]:
        classe = Y == c
        suspeito = adjacencia[classe, :][:, np.invert(classe)]
        fronteira.append(np.sum(suspeito, axis = 1) > 0)
    return np.concatenate(fronteira, axis = 0)

def overlap(X, Y, adjacencia):
    overlap = []
    u, _ = np.unique(Y, return_index = True)
    for c in u[np.argsort(_)]:
        classe = Y == c
        vizinho = adjacencia[classe, :][:, classe]
        suspeito = adjacencia[classe, :][:, np.invert(classe)]
        vizinho = np.sum(vizinho, axis = 1)
        suspeito = np.sum(suspeito, axis = 1)
        overlap.append(suspeito >= vizinho)
    overlap = np.concatenate(overlap, axis = 0)

    X = X[np.invert(overlap), :]
    Y = Y[np.invert(overlap)]
    adjacencia = lara_graph(X)
    return X, Y, adjacencia, overlap

def f(x, y):
    return classificador(np.array([x, y]))

def classificador(x):
    distancias = []
    labels = []
    for i in range(len(pares)):
        w = W[i]
        b = bias[i]
        s = sinal[i]
        decisor = np.dot(w, x) - b
        
        if decisor > 0:
            parcial = s
        else:
            parcial = -1 * s
        
        d = np.linalg.norm(x - medios[i])
        distancias.append(d)
        labels.append(parcial * d)
    
    slabels = np.sign(labels)
    menor = np.argmin(distancias)
    labels = 1 / np.abs(labels)
    labels = labels / np.sum(labels)
    labels = np.sign(labels) * slabels
    
    if labels[menor] > 0:
        return 1
    else:
        return -1

# read data
# data = pd.read_csv('data/raw.csv', header = None).values
# gg = pd.read_csv('data/raw-gg.csv', header = None).values
data = pd.read_csv('data/overlap.csv', header = None).values

X = data[:, 0:-1]
Y = data[:, -1]

# gg
adjacencia = lara_graph(X)
# D = distance_matrix(X, X) ** 2
# D[np.diag_indices(D.shape[0])] = 1e6

# n = X.shape[0]
# adjacencia = np.zeros(shape = (n, n))
# for i in range(n-1):
#     for j in range(i+1, n):
#         # print(i, j)
#         minimo = min(D[i, :] + D[j, :])
#         if (D[i, j] <= minimo):
#             adjacencia[i, j] = 1
#             adjacencia[j, i] = 1

# overlap
X, Y, adjacencia, _ = overlap(X, Y, adjacencia)
# overlap = []
# u, _ = np.unique(Y, return_index = True)
# for c in u[np.argsort(_)]:
#     classe = Y == c
#     vizinho = adjacencia[classe, :][:, classe]
#     suspeito = adjacencia[classe, :][:, np.invert(classe)]
#     vizinho = np.sum(vizinho, axis = 1)
#     suspeito = np.sum(suspeito, axis = 1)
#     overlap.append(suspeito >= vizinho)
# overlap = np.concatenate(overlap, axis = 0)

# X = X[np.invert(overlap), :]
# Y = Y[np.invert(overlap)]
# adjacencia = lara_graph(X)

# fronteira
fronteira = suporte(X, Y, adjacencia)
# fronteira = []
# u, _ = np.unique(Y, return_index = True)
# for c in u[np.argsort(_)]:
#     classe = Y == c
#     suspeito = adjacencia[classe, :][:, np.invert(classe)]
#     fronteira.append(np.sum(suspeito, axis = 1) > 0)
# fronteira = np.concatenate(fronteira, axis = 0)

# plot
for classe in np.unique(Y):
    idx = Y == classe
    plt.scatter(X[idx, 0], X[idx, 1], c = {1: 'red', -1: 'blue'}[classe])
plt.scatter(X[fronteira, 0], X[fronteira, 1], c = 'green')

# pares
u, _ = np.unique(Y, return_index = True)
classe = Y == u[np.argsort(_)][0]
cfronteira = np.copy(fronteira)
cfronteira[np.invert(classe)] = False
vertices = np.copy(adjacencia)
vertices[np.invert(cfronteira), :] = False
vertices[:, classe] = False
pares = np.where(vertices == 1)
pares = list(zip(pares[0], pares[1]))

# pesos
medios = []
W = []
bias = []
sinal = []
for par in pares:
    c = X[par[0]]
    d = X[par[1]]
    m = (c + d) / 2
    medios.append(m)
    w = c - d
    W.append(w)
    bias.append(np.dot(m, w))
    sinal.append(Y[par[0]])

# classificacao
yhat = []
for x in X:
    # distancias = []
    # labels = []
    # for i in range(len(pares)):
    #     w = W[i]
    #     b = bias[i]
    #     s = sinal[i]
    #     decisor = np.dot(w, x) - b
        
    #     if decisor > 0:
    #         parcial = s
    #     else:
    #         parcial = -1 * s
        
    #     d = np.linalg.norm(x - medios[i])
    #     distancias.append(d)
    #     labels.append(parcial * d)
    
    # slabels = np.sign(labels)
    # menor = np.argmin(distancias)
    # labels = 1 / np.abs(labels)
    # labels = labels / np.sum(labels)
    # labels = np.sign(labels) * slabels
    
    # if labels[menor] > 0:
    #     yhat.append(1)
    # else:
    #     yhat.append(-1)
    yhat.append(classificador(x))

# contorno
N = 50
xx = np.linspace(2, 8, N)
yy = np.linspace(2, 8, N)
XX, YY = np.meshgrid(xx, yy)
Z = np.zeros(shape = XX.shape)
for i in range(N):
    for j in range(N):
        x = XX[i, j]
        y = YY[i, j]
        Z[i, j] = f(x, y)
plt.contour(XX, YY, Z, [-1, 1])
