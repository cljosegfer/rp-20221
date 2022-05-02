#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:54:47 2022

@author: jose
"""

# sim, eu sou q isso esta horrivel

import numpy as np
from scipy.spatial import distance_matrix

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

def pares(adjacencia, fronteira, Y):
    u, _ = np.unique(Y, return_index = True)
    classe = Y == u[np.argsort(_)][0]
    cfronteira = np.copy(fronteira)
    cfronteira[np.invert(classe)] = False
    vertices = np.copy(adjacencia)
    vertices[np.invert(cfronteira), :] = False
    vertices[:, classe] = False
    pares = np.where(vertices == 1)
    pares = list(zip(pares[0], pares[1]))
    return pares

def pesos(X, Y, pares):
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
    return medios, W, bias, sinal

def classificador(x, pares, medios, W, bias, sinal):
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

def f(x, y, pares, medios, W, bias, sinal):
    return classificador(np.array([x, y]), pares, medios, W, bias, sinal)
