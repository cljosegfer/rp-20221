#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 09:15:15 2022

@author: jose
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('placas.jpg', 0)
target = 'target0.jpg'
template = cv.imread(target, 0)

W, H = img.shape[::-1]
w, h = template.shape[::-1]

img = np.array(img, dtype = np.int8)
template = np.array(template, dtype = np.int8)
cost = np.ones(shape = img.shape) * 1e6

minimo = 1e6
best = None
for i in range(H - h + 1):
    for j in range(W - w + 1):
        window = img[i:(i + h), j:(j + w)]
        norm = np.linalg.norm(window - template)
        cost[i:(i + h), j:(j + w)] = np.minimum(norm, cost[i:(i + h), j:(j + w)])
        
        if norm < minimo:
            minimo = norm
            best = window
            
            top_left = (j, i)
            bottom_right = (j + w, i + h)
best = np.array(best, dtype = np.uint8)

# trace
img = np.array(img, dtype = np.uint8)
cv.rectangle(img, top_left, bottom_right, 255, 2)

# cost
cost *= (255.0/cost.max())
cost = np.array(cost, dtype = np.uint8)

# plot
plt.figure(figsize = (20, 6), dpi = 80)
plt.subplot(121),plt.imshow(cost, cmap = 'gray')
plt.title('Superfície de Custo'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img, cmap = 'gray')
plt.title('Região Encontrada'), plt.xticks([]), plt.yticks([])
plt.savefig('fig/result-' + target[:-4] + '.png')
