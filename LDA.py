
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
    center class matrices Zi
    finding Sb (replacement to B for higher no of classes) in LDA  (done)
    finding mean for each class training data (done)
# =============================================================================
'''
import numpy as np
#number of images in total
rows=400
#differenct people
number_of_classes=40
nImages_in_each_class=10
#dimentions of vector (image) before reduction
dimentions=10304

#making dummy matrix to play with
#D=np.random.rand(400,10304)
#TODO repalce this with real D
D=np.ones((rows, dimentions))
#print(D)

z=np.zeros((1, dimentions))
#making an array to save means of each class
classes_means=np.zeros((number_of_classes,dimentions))
for i in range(number_of_classes):
    for j in range (nImages_in_each_class):
        #adding evens because index of arrays start at zero not 1
        if(j % 2 == 0):
            classes_means[i]+=D[i*10+j][:]
classes_means/=(nImages_in_each_class/2)
    

#finding Sb (replacement to B for higher no of classes) in LDA
#nk is number of samples in kth class
nk=nImages_in_each_class /2 
#overall sample mean
#get mean of each column and result is 1 row and 10304(dimentions) columns
meu=np.mean(D,axis=0)
#print("meu:")
#print(meu.shape) 
#initializing Sb
Sb=np.zeros((dimentions,dimentions))

for k in range(number_of_classes):
    diff_means=classes_means[k]-meu
    diff_means=np.reshape(diff_means,(1,10304) )
    dm_t=diff_means.transpose()
    B=np.matmul(dm_t,diff_means)
    Sb+=nk*B
#print(Sb.shape) #(10304, 10304)
#center class matrices Zi i=0,1,2...39
Z=D
for i in range(number_of_classes):
    for j in range (nImages_in_each_class):
        #TODO classes means for dummy data is array of 5s is that correct?
        if(j % 2 == 0):
            #print(classes_means[i])
            #initializing S : within-class scatter matrix
            S=Sb=np.zeros((1,dimentions))
            #class scatter matrices Si
            Z[i*10+j]-=classes_means[i][:]
#TODO check if Si the within class scatter matrix is same as covarience matrix
#of Z center class matrix
# or is it equal to covarience matrix of class i mean and so on
#or should I calculate S as in Algorithm
S_i=np.cov(classes_means[i])
S=np.cov(Z.T)
print(S.shape)

#print(Z)
#print(Z.shape) #(400, 10304)
#print(S.shape) #(1, 10304)
#Sb replaces Si


