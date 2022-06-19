#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 10:51:02 2022

@author: jose
"""

from scipy import io
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score

# params
indice = 'silhouette'
dir_path = 'data'
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

metodos = ('base', 
           'lmnn', 
           'lfda')

report = np.zeros(shape = (len(datasets), len(metodos)))

for mt, metodo in enumerate(metodos):
    for ds, dataset in enumerate(datasets):
        scores = []
        fold_n = 0
        print('metodo: {} dataset: {} / {} fold: {} / {}'.format(metodo, datasets.index(dataset) + 1, 
                                                      len(datasets), fold_n + 1, K))
        
        # read
        filename = '{}/{}/exportBase_{}_folds_10_exec_{}.mat'.format(
            dir_path, metodo, dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        # train / test
        train = data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        
        # davies-bouldin
        X = np.concatenate((train, test), axis = 0)
        y = np.concatenate((classTrain, classTest), axis = 0)
        score = silhouette_score(X = X, labels = y)
        scores.append(score)
        
        # report
        report[ds, mt] = np.around(np.mean(scores), decimals = 2)

# export
relatorio = pd.DataFrame(report, columns = metodos, index = datasets)
relatorio.to_csv('output/indices/{}.csv'.format(indice))
