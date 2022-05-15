#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 09:57:16 2022

@author: jose
"""

import numpy as np
import pandas as pd

# read
metodo = 'base'
auc_b = pd.read_csv('output/{}/coletivo.csv'.format(metodo))
dis_b = pd.read_csv('output/{}/d_coletivo.csv'.format(metodo))
silh = pd.read_csv('output/otm-log.csv', header = None).drop([0], axis = 1)

metodo = 'silh'
auc_s = pd.read_csv('output/{}/coletivo.csv'.format(metodo))
dis_s = pd.read_csv('output/{}/d_coletivo.csv'.format(metodo)).drop(['Unnamed: 0'], axis = 1)

# auc
log = []
for (_, s1), (_, s2) in zip(auc_b.iterrows(), auc_s.iterrows()):
    dataset = s1['Unnamed: 0']
    auc_mu = round(s1['mean'], 2)
    auc_sd = round(s1['sd'], 2)
    auc_mu1 = round(s2['mean'], 2)
    auc_sd1 = round(s2['sd'], 2)
    string = r'{} & ${} \pm {}$ & ${} \pm {}$ \\'.format(dataset, auc_mu, auc_sd, auc_mu1, auc_sd1)
    log.append(string)
    log.append('\hline')
np.savetxt('output/table/auc.csv', log, fmt = '%s')

# silh
log = []
for (_, s1), (_, s3) in zip(auc_b.iterrows(), silh.iterrows()):
    dataset = s1['Unnamed: 0']
    score = round(s3[1], 2)
    score_mu = round(s3[2], 2)
    score_sd = round(s3[3], 2)
    string = r'{} & ${}$ & ${} \pm {}$ \\'.format(dataset, score, score_mu, score_sd)
    log.append(string)
    log.append('\hline')
np.savetxt('output/table/silh.csv', log, fmt = '%s')

# dis
log = []
for (_, s1), (_, s2) in zip(dis_b.iterrows(), dis_s.iterrows()):
    dataset = s1['Unnamed: 0']
    auc_mu = round(s1['mean'], 2)
    auc_sd = round(s1['sd'], 2)
    auc_mu1 = round(s2['mean'], 2)
    auc_sd1 = round(s2['sd'], 2)
    string = r'{} & ${} \pm {}$ & ${} \pm {}$ \\'.format(dataset, auc_mu, auc_sd, auc_mu1, auc_sd1)
    log.append(string)
    log.append('\hline')
np.savetxt('output/table/dis.csv', log, fmt = '%s')
