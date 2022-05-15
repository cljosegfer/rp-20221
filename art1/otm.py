#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:27:28 2022

@author: jose
"""

import numpy as np
import pandas as pd
from scipy import io
from sklearn.metrics import silhouette_score
from scipy import optimize

def silh(L):
    return silhouette_score(X = np.dot(L, train.T).T, labels = classTrain)

def f(L):
    n = int(np.sqrt(len(L)))
    return (-1) * silh(L.reshape(n, -1))

def g(L):
    n = int(np.sqrt(len(L)))
    L = L.reshape(n, -1)
    M = np.dot(L, L.T)
    return np.all(np.linalg.eigvals(M) > 0) * 1

# params
env_path = 'data/temp_bases/temp'
trans_path = 'data/trans'
datasets = ('australian', 
            'banknote', 
            'breastcancer', 
            'breastHess', 
            'bupa', 
            'climate', 
            'diabetes', 
            'fertility', 
            'german', 
            'golub', 
            'haberman', 
            'heart', 
            'ILPD', 
            'parkinsons', 
            'sonar')
K = 10
log = []

# processo
# dataset = 'breastHess'
for dataset in datasets:
    old = []
    score = []
    # fold_n = 0
    for fold_n in range(K):
        print('dataset: {} / {} fold: {} / {}'.format(datasets.index(dataset) + 1, len(datasets), fold_n + 1, K))
        
        # read
        filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
            env_path, dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        # train / test
        train = data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        
        # otm
        n = len(train[0, :])
        L0 = np.random.randn(n, n).flatten()
        # res = optimize.minimize(f, L0, constraints = {'fun': g, 'type':'ineq'}, 
        #                         options = {'maxiter': 1})
        res = optimize.minimize(f, L0, constraints = {'fun': g, 'type':'ineq'})
        L = res.x.reshape(n, -1)
        # L = L0.reshape(n, -1)
        
        # transform
        data_mat['data']['train'][0][0] = np.dot(L, train.T).T
        data_mat['data']['test'][0][0] = np.dot(L, test.T).T
        
        # export
        export_filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
                    trans_path, dataset, fold_n + 1)
        io.savemat(export_filename, data_mat)
        
        # log
        X = np.concatenate((train, test), axis = 0)
        y = np.concatenate((classTrain, classTest), axis = 0)
        old.append(silhouette_score(X = X, labels = y))
        score.append(silhouette_score(X = np.dot(L, X.T).T, labels = y))
    log.append([dataset, np.mean(old), np.mean(score), np.std(score)])
np.savetxt('output/otm-log.csv', log, delimiter = ',', fmt = '%s')
