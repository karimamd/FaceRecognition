#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:34:52 2018

@author: kareem
"""
import glob
import numpy as np
import cv2
X_data = []
files = glob.glob ("*.pgm")
for myFile in files:
    image = cv2.imread (myFile)
    X_data.append (image)

print('X_data shape:', np.array(X_data).shape())
