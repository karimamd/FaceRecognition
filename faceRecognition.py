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
import pickle
from pathlib import Path
from scipy.spatial import distance


# =============================================================================
# SIMPLE CLASSIFIER ROUTINE
# =============================================================================


def classify(
    data_mat0,
    data_mat1,
    label_mat0,
    label_mat1,
    ):
    
    length = data_mat0.shape[0]
    map_ = np.zeros((1, length))
    label_mat_new = np.zeros((1, length))
    accuracy_mat = np.ones((1, length))

    for i in range(0, length):
        for j in range(0, length):
            map_[0, j] = distance.euclidean(data_mat1[i], data_mat0[j])
        arg = map_.argmin()
        label_mat_new[0, i] = label_mat0[arg]
        if label_mat_new[0, i] != label_mat1[i]:
            accuracy_mat[0, i] = 0

    return 100 * np.sum(accuracy_mat) / np.size(accuracy_mat)


# =============================================================================
# PROJECTION MATRIX ROUTINE
# =============================================================================

def getProj(data_matrix, alpha, str):
    
    isAlpha = False
    number = 0
    data_matrix_centered = data_matrix - np.mean(data_matrix, axis=0)
    data_matrix_cov = np.cov(data_matrix_centered, rowvar=False)
    (eigenValues, eigenVectors) = LA.eigh(data_matrix_cov)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    total = np.sum(eigenValues)

    while isAlpha == False:
        sum_eigVal = 0.0
        for i in range(eigenValues.size):
            sum_eigVal = sum_eigVal + eigenValues[i] / total
            number += 1
            if math.isclose(sum_eigVal, alpha) or sum_eigVal > alpha:
                isAlpha = True
                break

    projection_matrix = np.matrix([eigenVectors[n] for n in
                                  range(number)]).T

    if not Path(str + '.pickle').exists():
        with open(str + '.pickle', 'wb') as handle:
            pickle.dump(projection_matrix, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print ('Error.' + ' Another file with same name already found.')
    return projection_matrix


# =============================================================================
# PCA ROUTINE
# =============================================================================

def PCA_(
    data_matrix,
    proj_matrix,
    alpha,
    str,
    ):

    data_matrix_centered = data_matrix - np.mean(data_matrix, axis=0)
    if proj_matrix is None:
        projection_matrix = getProj(data_matrix, alpha, str)
    else:
        projection_matrix = proj_matrix

    rd_data_matrix = np.matmul(data_matrix_centered, projection_matrix)
    return rd_data_matrix


# =============================================================================
# INITIAL SETUP
# =============================================================================

imgMat = np.zeros((0, 10304))
temp = np.arange(1, 41, 1)
label_matrix = np.array([[temp[i]] * 10 for i in range(temp.size)])
label_matrix = label_matrix.flatten()
folder = 'C:\\FaceReco\\orl_faces\\'
for j in range(1, 41):
    direction = folder + 's' + str(j) + '\\'
    for i in range(1, 11):
        directory = direction + str(i) + '.pgm'
        image = img.imread(directory).T
        imageVect = np.asmatrix(image.flatten())
        imgMat = np.concatenate((imgMat, imageVect))

test_data_matrix = imgMat[::1]
training_data_matrix = imgMat[::2]

label_test = label_matrix[::1]
label_training = label_matrix[::2]

# =============================================================================
# COMPUTING ACCURACY FOR EACH ALPHA
# =============================================================================

# proj_data_mat=getProj(training_data_matrix,0.95,"proj_data_mat_0.95")

alpha = np.matrix([[0.8, 0.85, 0.9, 0.95]])

for k in range(alpha.size):
    with open('proj_data_mat_' + str(alpha[0, k]) + '.pickle', 'rb') as \
        handle:
        proj_data_mat = pickle.load(handle)
    training_data_matrix_rd = PCA_(training_data_matrix, proj_data_mat,
                                   alpha[0, k], '')
    test_data_matrix_rd = PCA_(test_data_matrix, proj_data_mat,
                               alpha[0, k], '')
    acc_prc = classify(training_data_matrix_rd, test_data_matrix_rd,
                       label_training, label_test)
    print ('For alpha: ' + str(alpha[0, k]) + ' accuracy percentage= ' \
        + str(acc_prc) + '%\n')
        

               
            
        
    
    