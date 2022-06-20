#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:23:56 2022

@author: jose
"""

import numpy as np
import pandas as pd

metodos = ('svm', 
           'clas', 
           'svm_composto', 
           'clas_composto')

# read
metodo = 'clas_composto'
data = pd.read_csv('output/lm/{}.csv'.format(metodo), index_col = 0)

# table
log = []
for _, row in data.iterrows():
    string = r'{0} & ${1:.2f}$ & ${2:.0e}$ & ${3:.2f}$ & ${4:.0e}$ & ${5:.2f}$ & ${6:.0e}$ \\'.format(row.name, 
                    row['base_coef'], row['base_pvalor'], 
                    row['lmnn_coef'], row['lmnn_pvalor'], 
                    row['lfda_coef'], row['lfda_pvalor'], )
    log.append(string)
    log.append('\hline')
np.savetxt('output/table.txt', log, fmt = '%s')