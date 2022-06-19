#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 16:50:00 2022

@author: jose
"""

from scipy import io
import pandas as pd
import numpy as np

# params
indice = 'q'
indice2 = 'discard'
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
report2 = np.zeros(shape = (len(datasets), len(metodos)))

# metodo = 'base'
# mt = 0
# dataset = 'australian'
# ds = 0

for mt, metodo in enumerate(metodos):
    for ds, dataset in enumerate(datasets):
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
        
        X = np.concatenate((train, test), axis = 0)
        y = np.concatenate((classTrain, classTest), axis = 0)
        
        # gg
        path = '{}/gg/{}/ggBase_{}_folds_10_exec_{}.csv'.format(
            dir_path, metodo, dataset, fold_n + 1)
        gg = pd.read_csv(path).to_numpy()
        
        # qualidade
        scores = []
        for i, row in enumerate(gg):
            vizinhos = np.where(row == 1)[0]
            
            degree = len(vizinhos)
            opposite = 0
            for vizinho in vizinhos:
                opposite += np.abs(y[0] - y[vizinho]) / 2
            q = 1 - opposite / degree
            scores.append(q)
        
        # report
        report[ds, mt] = np.around(np.mean(scores), decimals = 2)
        
        # descarte
        classes = np.unique(y)
        scores = np.array(scores)
        discard = 0
        for classe in classes:
            cluster = y == classe
            tc = np.mean(scores[cluster])
            filtro = scores[cluster] < tc
            discard += np.sum(filtro) / len(scores)
        
        # report
        report2[ds, mt] = np.around(discard, decimals = 2)

# export
relatorio = pd.DataFrame(report, columns = metodos)
relatorio.to_csv('output/indices/{}.csv'.format(indice), index = datasets)
relatorio = pd.DataFrame(report2, columns = metodos)
relatorio.to_csv('output/indices/{}.csv'.format(indice2), index = datasets)
