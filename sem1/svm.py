#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 01:05:05 2022

@author: jose
"""

from scipy import io
import pandas as pd
import numpy as np
from sklearn import svm

# params
env_path = 'data/temp_bases/temp'
gg_path = 'data/gg'
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

# dataset = 'breastHess'
for dataset in datasets:
    size = []
    sup = []
    edit = []
    ratio = []
# fold_n = 0
    for fold_n in range(K):

        # print('dataset: {} / {} fold: {} / {}'.format(datasets.index(dataset) + 1, 
        #                                               len(datasets), fold_n + 1, K))
        
        # read
        filename = '{}/exportBase_{}_folds_10_exec_{}.mat'.format(
            env_path, dataset, fold_n + 1)
        data_mat = io.loadmat(filename)
        
        # train / test
        train = data_mat['data']['train'][0][0]
        classTrain = data_mat['data']['classTrain'][0][0].ravel()
        test = data_mat['data']['test'][0][0]
        classTest = data_mat['data']['classTest'][0][0].ravel()
        
        # gg
        path = '{}/ggBase_{}_folds_10_exec_{}.csv'.format(
            gg_path, dataset, fold_n + 1)
        gg = pd.read_csv(path).to_numpy()
        
        # svm
        clf = svm.SVC()
        clf.fit(train, classTrain)
        
        # support
        suporte = np.sort(clf.support_) + 1
        coincide = sum(point in suporte for point in gg)
        relacao = coincide / len(suporte)
        
        # log
        size.append(len(train))
        sup.append(len(suporte))
        edit.append(len(gg))
        ratio.append(relacao)
    print('{},{},{},{},{}'.format(dataset, np.mean(size), np.mean(sup), np.mean(edit), np.mean(ratio)))
