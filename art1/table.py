#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 09:26:46 2022

@author: jose
"""

import numpy as np
import pandas as pd

# read
metodo = 'silh'
auc = pd.read_csv('output/{}/coletivo.csv'.format(metodo))
dis = pd.read_csv('output/{}/d_coletivo.csv'.format(metodo)).drop(['Unnamed: 0'], axis = 1)
silh = pd.read_csv('output/otm-log.csv', header = None).drop([0], axis = 1)

# table
log = []
for (_, s1), (_, s2), (_, s3) in zip(auc.iterrows(), dis.iterrows(), silh.iterrows()):
    dataset = s1['Unnamed: 0']
    auc_mu = round(s1['mean'], 2)
    auc_sd = round(s1['sd'], 2)
    score = round(s3[1], 2)
    dis_mu = round(s2['mean'], 2)
    dis_sd = round(s2['sd'], 2)
    string = r'{} & ${} \pm {}$ & ${}$ & ${} \pm {}$ \\'.format(dataset, auc_mu, auc_sd, score, dis_mu, dis_sd)
    if metodo == 'silh':
        score_mu = round(s3[2], 2)
        score_sd = round(s3[3], 2)
        string = r'{} & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ \\'.format(dataset, auc_mu, auc_sd, score_mu, score_sd, dis_mu, dis_sd)
    log.append(string)
    log.append('\hline')
np.savetxt('output/table/table.csv', log, fmt = '%s')
