#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 09:51:13 2018

@author: Zyad Shokry & Kareem Abd-El-Salam
"""

import matplotlib.image as img
import numpy as np
import math
from numpy import linalg as LA

folder="C:\\FaceReco\\orl_faces\\"

imgMat=np.zeros((0,10304))
temp=np.arange(1,41,1)
label_matrix=np.array([[temp[i]]*10 for i in range (temp.size)])
label_matrix=label_matrix.flatten()

def PCA_(data_matrix,alpha):
    isAlpha=False
    number=0
    data_matrix_centered= data_matrix - np.mean(data_matrix,axis=0)
    data_matrix_cov= np.cov(data_matrix_centered,rowvar=False)
    eigenValues, eigenVectors = LA.eigh(data_matrix_cov)
    idx = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    total=np.sum(eigenValues)
#    print(eigenValues, " eigVal\n",eigenVectors," eigVec\n")
    while(isAlpha==False):
        sum_eigVal=0.0
        for i in range (eigenValues.size):
            sum_eigVal=sum_eigVal+eigenValues[i]/total
            number+=1
            if(sum_eigVal>=alpha):
                isAlpha=True
                break
            
    projection_matrix=np.matrix([eigenVectors[n] for n in range (number)]).T
    print(projection_matrix,number)
    rd_data_matrix= np.matmul(data_matrix,projection_matrix)
    return rd_data_matrix

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
        
training_data_matrix=imgMat[::2]
test_data_matrxi=imgMat[::1]
        

               
            
        
    
    