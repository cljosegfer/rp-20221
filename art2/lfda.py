#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 08:47:40 2022

@author: jose
"""

import scipy
import pandas as pd
import numpy as np

from metric_learn import LFDA

# params
metodo = 'lfda'
dir_path = 'data/base'
out_path = 'data/{}'.format(metodo)
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

for dataset in datasets:
    for fold_n in range(K):
        print('dataset: {} / {} fold: {} / {}'.format(datasets.index(dataset) + 1, 
                                                      len(datasets), fold_n + 1, K))
        
        # read
        filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
            dir_path, dataset, fold_n + 1)
        data_mat = scipy.io.loadmat(filename)
        
        # train / test
        train = data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        
        # model
        model = LFDA(n_components = train.shape[1] - 1)
        model.fit(train, classTrain)
        
        # transform
        data_mat['data']['train'][0][0] = model.transform(train)
        data_mat['data']['test'][0][0] = model.transform(test)
        
        # export
        export_filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
            out_path, dataset, fold_n + 1)
        scipy.io.savemat(export_filename, data_mat)
