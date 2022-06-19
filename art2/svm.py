#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 09:47:41 2022

@author: jose
"""

from scipy import io
import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

# params
dir_path = 'data'
out_path = 'output'
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

report = np.zeros(shape = (len(datasets), 2*len(metodos)))
for mt, metodo in enumerate(metodos):
    for ds, dataset in enumerate(datasets):
        acuracia = []
        for fold_n in range(K):
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
            
            # model
            model = SVC()
            model.fit(train, classTrain)
            
            # eval
            # score = model.score(test, classTest)
            score = roc_auc_score(classTest, model.predict(test))
            acuracia.append(score)    
        report[ds, 2*mt] = np.around(np.mean(acuracia), decimals = 2)
        report[ds, 2*mt + 1] = np.around(np.std(acuracia), decimals = 2)

# export
columns = []
for metodo in metodos:
    columns.append('{}_mean'.format(metodo))
    columns.append('{}_std'.format(metodo))
relatorio = pd.DataFrame(report, columns = columns, index = datasets)
relatorio.to_csv('output/svm.csv')
