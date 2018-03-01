
'''
Dimentionality reduction and classification using LDA
# =============================================================================
# every class is 10 vectors each of r values (supposidly 10304) because LDA
will not continue from where PCA left

#or each class is 10 rows in the data matrix D
#required is to take mean of each 10 rows into 1 row and 10304 (r) columns

#but they are not 10 only 5 because other 5 are test data

strategy:
    get line (direction) with highest eigen value after implementing LDA
    fully
    project all points or only mean of each class on line
    
    mean is not rational if Knn : if K is > 1
    
    so  we will project test point on line now dimentions is much reduced and 
    can measure distance with all training points and find label accordingly
    
current step:
    finding Sb (replacement to B for higher no of classes) in LDA
    finding mean for each class training data (done)
# =============================================================================
'''
import numpy as np
#number of images in total
rows=400
#differenct people
number_of_classes=40
nimages_in_each_class=10
#dimentions of vector (image) before reduction
dimentions=10304

#making dummy matrix to play with
#D=np.random.rand(400,10304)
#TODO repalce this with real D
D=np.ones((rows, dimentions))
#print(D)
M1=np.mean(D,axis=1)
#print(M1)
z=np.zeros((1, dimentions))
#making an array to save means of each class
classes_means=np.zeros((number_of_classes,dimentions))
for i in range(number_of_classes):
    for j in range (nimages_in_each_class):
        #adding evens because index of arrays start at zero not 1
        if(j % 2 == 0):
            classes_means[i]+=D[i*10+j][:]
    

