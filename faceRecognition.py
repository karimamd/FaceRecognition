#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:51:13 2018

@author: kareem
"""

import matplotlib.image as img
import numpy as np
folder="C:\\FaceReco\\orl_faces\\"
imgMat=np.zeros((0,10304))

#imgMat=np.concatenate((imgMat,imgMat2),axis=0)
#print(imgMat)
#
for j in range(1,41):
    direction=folder+"s"+str(j)+"\\"
    for i in range (1,11):
        directory=direction+str(i)+ ".pgm"
       # print(directory)
        image = (img.imread(directory)).T
        imageVect=np.asmatrix(image.flatten())
        imgMat=np.concatenate((imgMat,imageVect))

        
