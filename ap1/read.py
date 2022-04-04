#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 18:01:14 2022

@author: jose
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('placas.jpg', 0)
target = 'samples/target3.png'
template = cv.imread(target, 0)
