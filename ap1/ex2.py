#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 14:06:14 2022

@author: jose
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import model.alpr as alpr
lpr = alpr.AutoLPR(decoder = 'bestPath', normalise=True)
lpr.load(crnn_path = 'model/weights/best-fyp-improved.pth')

from difflib import SequenceMatcher
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

norma = 'CNN'

targets = ['4BEX972', '1890', 'JUX580', '763FYK']
# target = 'JUX580'

for target in targets:
    img = cv.imread('placas.jpg', 0)
    template = np.zeros(shape = (100, 100), dtype = np.int8)

    W, H = img.shape[::-1]
    w, h = template.shape[::-1]

    img = np.array(img, dtype = np.int8)
    cost = np.ones(shape = img.shape) * 1e6

    minimo = 1e6
    best = None
    print(target)
    for i in tqdm(range(H - h + 1)):
        for j in range(W - w + 1):
            window = img[i:(i + h), j:(j + w)]
            if norma == 'CNN':
                window = np.array(window, dtype = np.uint8)
                cv.imwrite('temp.png', window)
                text = lpr.predict('temp.png')
                # text = lpr.predict(window)
                norm = 1 - similar(text, target)
                # print(text, norm)
            # elif norma == 'L1':
            #     norm = np.linalg.norm(window - template, ord = 1)
            # else:
            #     norm = np.linalg.norm(window - template)
            cost[i:(i + h), j:(j + w)] = np.minimum(norm, cost[i:(i + h), j:(j + w)])
            
            if norm < minimo:
                minimo = norm
                best = window
                
                top_left = (j, i)
                bottom_right = (j + w, i + h)
    best = np.array(best, dtype = np.uint8)

    print(top_left, bottom_right)
    print(minimo)
    cv.imwrite('best.png', best)
    np.save('cost.npy', cost)

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
    plt.suptitle('Métrica: ' + norma)
    plt.savefig('fig/' + norma + '-' + target[:-4] + '.png')
