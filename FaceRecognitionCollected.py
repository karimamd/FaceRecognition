#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:03:05 2018

@author: kareem
"""
#initial imports
import matplotlib.image as img
import numpy as np
import math
from numpy import linalg as LA
import pickle
from pathlib import Path
from scipy.spatial import distance
import operator
import matplotlib.pyplot as plt

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
#put folder name containing ORL dataset
#note that this is for linux directory system for windows use those lines with changes
#folder="C:\\FaceReco\\orl_faces\\"
# direction=folder+"s"+str(j)+"\\"
folder = 'orl_faces/'
for j in range(1, 41):
    direction = folder + 's' + str(j) + '/'
    for i in range(1, 11):
        directory = direction + str(i) + '.pgm'
        image = img.imread(directory).T
        imageVect = np.asmatrix(image.flatten())
        imgMat = np.concatenate((imgMat, imageVect))

test_data_matrix = imgMat[1::1]
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

print("image matrix" ,imgMat)
print(imgMat[0:5])

#number of images in total
rows=400
#differenct people
number_of_classes=40
nImages_in_each_class=10
#dimentions of vector (image) before reduction
dimentions=10304
D=imgMat

#making an array to save means of each class (40,10304)
classes_means=np.zeros((number_of_classes,dimentions))

for i in range(number_of_classes):
    #adding evens because index of arrays start at zero not 1
    classes_means[i]=np.mean(test_data_matrix[i*5:i*5+5],axis=0)

#finding Sb (replacement to B for higher no of classes) in LDA
#nk is number of samples in kth class
nk=nImages_in_each_class // 2 
#overall sample mean
#get mean of each column and result is 1 row and 10304(dimentions) columns
meu=np.mean(D,axis=0)
#initializing Sb
Sb=np.zeros((dimentions,dimentions))

for k in range(number_of_classes):
    diff_means=classes_means[k]-meu
    diff_means=np.reshape(diff_means,(1,10304) )
    dm_t=diff_means.transpose()
    B=np.matmul(dm_t,diff_means)
    Sb+=nk*B
#print(Sb.shape) #(10304, 10304)

"""center class matrices Zi i=0,1,2...39 : """
Z=D
for i in range(number_of_classes):
    for j in range (nImages_in_each_class):
        if(j % 2 == 0):
            Z[i*10+j]-=classes_means[i][:]
print("shape of Z:",Z.shape) #(400, 10304)

"""within class scatter matrix S :"""
          
#TODO check if Si the within class scatter matrix is same as covarience matrix
#of Z center class matrix
# or is it equal to covarience matrix of class i mean and so on
#or should I calculate S as in Algorithm
#S_i=np.cov(classes_means[i]) #S+=S_i #in for loop ?
#or : #S=np.cov(Z.T) #produces 10304 * 10304 matrix
#print(S.shape)
#S_trail=np.zeros((1,dimentions))
            

#TODO check if it is true to make Zi as 5 samples for each i and summing them
#or is this the thing that made us say that Si was covariance of sth in the first place? 
S=np.zeros((dimentions,dimentions))
S_initial=np.zeros((nImages_in_each_class // 2,dimentions))
for i in range (number_of_classes):
    for j in range (nImages_in_each_class // 1):
        if (j % 2== 0):    
            S_i=S_initial
            S_i[j // 2]= S_i[j // 2] +  Z[i*10+j]
    S_i=np.dot(S_i.T,S_i)
    S+=S_i

#S_inv=np.linalg.inv(S)
#S_inv= np.linalg.solve(S, np.identity(10304))
S_inv= np.linalg.pinv(S)

#Eigen vectors and values:
S_inv_mul_B=np.matmul(S_inv,Sb)
print("shapes of S_inv and Sb,S_inv_mul_",S_inv.shape,Sb.shape,S_inv_mul_B.shape)

#commenting those because of high processing

eigenvals,eigenvecs = np.linalg.eig(S_inv_mul_B)
#sorting eigen values in descending order
sort_idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[sort_idx]
eigenvecs = eigenvecs[:,sort_idx]
print("shape of eigen values",eigenvals)
print("shape of eigen vectors",eigenvecs)

lamb=np.zeros(39)
lamp=eigenvals[:39]
u=u.T

p_train=(u.transpose().dot(training_data_matrix))/(u.transpose().dot(u)) *u    
p_test=(u.transpose().dot(test_data_matrix))/(u.transpose().dot(u)) *u
