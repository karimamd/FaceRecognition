#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:51:13 2018

@author: kareem
"""

import matplotlib.image as img
import numpy as np
folder="/home/kareem/Study/Term 8 Local/Pattern Recognition/Assignments/orl_faces/"
imgMat=np.ones((10304,))

#imgMat=np.concatenate((imgMat,imgMat2),axis=0)
#print(imgMat)
#
for j in range(1,41):
    direction=folder+"s"+str(j)+"/"
    for i in range (1,11):
        directory=direction+str(i)+ ".pgm"
       # print(directory)
        image = (img.imread(directory)).T
       # print(image.shape)
        imageVect=image.flatten()
       # print(imgMat.shape)
        #print(imageVect.shape)
        imgMat = np.expand_dims(imgMat, axis=1)
        imgMat=np.append(imgMat,imageVect,axis=1)


print(imgMat.shape)
#print(image.shape)
        