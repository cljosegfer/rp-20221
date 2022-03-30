#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 14:11:31 2022

@author: jose
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('placas.jpg', 0)
target = 'target1.jpg'
template = cv.imread(target, 0)
w, h = template.shape[::-1]

meth = 'cv.TM_CCOEFF_NORMED'
method = eval(meth)

# template match
res = cv.matchTemplate(img, template, method)
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv.rectangle(img, top_left, bottom_right, 255, 2)

# plot
plt.figure(figsize = (20, 6), dpi = 80)
plt.subplot(121),plt.imshow(res, cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img, cmap = 'gray')
plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)
plt.savefig('fig/opencv-' + target[:-4] + '.png')


