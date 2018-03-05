#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:11:12 2018

@author: M.Mohy
"""

import numpy as np
import operator
import faceRecognition
import matplotlib.pyplot as plt

count=1
labelVector =[]
for x in range(40): #constructing the classes of each instance
    for j in range(5):
        labelVector.append(count)
    count= count+1

def getAccuracy(predictedClasses, label_vector): # check predicted classes for correctness
	correct = 0
	for x in range(len(predictedClasses)):
		if predictedClasses[x] is label_vector[x]:
			correct += 1
	return (correct/float(len(predictedClasses))) * 100.0
def Knn(trainingSet,label_vector,test,k):
    distances =[]
    for x in range(len(trainingSet)): #calculating distance between test instance and all training data 
        distance = np.linalg.norm(np.subtract(test,trainingSet[x]))
        distances.append((label_vector[x],distance)) #storung the distance with the class of the training instance 
    distances.sort(key=operator.itemgetter(1)) #sorting distances array by the distance
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x]) #get the K neighbors of the instance
    classes ={}
    for x in range(len(neighbors)): #check classes for the neighbors
        response= neighbors[x][0] 
        if response in classes:
            classes[response] += 1 #classVote found! increase voting by 1
        else:
            classes[response] = 1 #new classVote! add it to the dict
    sortedVotes = sorted(classes.items(), key=operator.itemgetter(1), reverse=True)#sort the class votes in decreasing order
    return sortedVotes[0][0] #return the most voted class number
#list for getting the predictions for all test instances
#for x in range(200):    #test all the test_data_matrix
#    list_of_predictions.append(Knn(faceRecognition.training_data_matrix,labelVector,faceRecognition.test_data_matrix[x],3))
#print(getAccuracy(list_of_predictions,labelVector),"%")
k=[1,3,5,7]
accuracy=[]
for i in range(4):
    list_of_predictions=[] 
    for x in range(200):    #test all the test_data_matrix
        list_of_predictions.append(Knn(faceRecognition.training_data_matrix,labelVector,faceRecognition.test_data_matrix[x],k[i]))
    accuracy.append(getAccuracy(list_of_predictions,labelVector))
    print(accuracy[i],"%")
plt.plot(k,accuracy)
plt.ylabel("accuracy in %")
plt.xlabel("K")
plt.show