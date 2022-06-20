#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:03:08 2022

@author: jose
"""

import numpy as np
import pandas as pd

indices = ('calinski_harabasz', 
           'davies_bouldin', 
           'silhouette', 
           'q', 
           'discard')

# read
indice = 'discard'
data = pd.read_csv('output/indices/{}.csv'.format(indice), index_col = 0)

# table
log = []
for _, row in data.iterrows():
    string = r'{} & ${}$ & ${}$ & ${}$ \\'.format(row.name, 
                    row['base'], row['lmnn'], row['lfda'])
    log.append(string)
    log.append('\hline')
np.savetxt('output/table.txt', log, fmt = '%s')
