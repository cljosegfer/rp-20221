#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:38:07 2022

@author: jose
"""

import numpy as np
import pandas as pd

metodos = ('svm', 
           'clas')

# read
metodo = 'svm'
data = pd.read_csv('output/{}.csv'.format(metodo), index_col = 0)

# table
log = []
for _, row in data.iterrows():
    string = r'{} & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ \\'.format(row.name, 
                    row['base_mean'], row['base_std'], row['lmnn_mean'], row['lmnn_std'], row['lfda_mean'], row['lfda_std'])
    log.append(string)
    log.append('\hline')
np.savetxt('output/table.txt', log, fmt = '%s')
