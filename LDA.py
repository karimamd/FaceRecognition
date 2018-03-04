
'''
Always check for TODOs before rapping up

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
    getting 39 dominant eigen vectors instead of just one
    eigen values and vectors and dominant one (done by ny tested)
    within class scatter matrix S (done with doubt I think can go wrong need to implement it in the other way too)
    center class matrices Zi (done)
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

#z=np.zeros((1, dimentions))
#making an array to save means of each class (40,10304)
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
#print("shape of meu :",meu.shape)  #(10304,)
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
        #TODO classes means for dummy data is array of 5s is that correct?
        if(j % 2 == 0):
            #print(classes_means[i])
            #initializing S : within-class scatter matrix
            #TODO check what are real dimentions of S and initialize it accordingly
            #S=Sb=np.zeros((1,dimentions))
            #class scatter matrices Si
            Z[i*10+j]-=classes_means[i][:]
#print(Z)
#print("shape of Z:",Z.shape) #(400, 10304)

  
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
S_initial=np.zeros((5,dimentions))
for i in range (number_of_classes):
    for j in range (nImages_in_each_class):
        if (j % 2== 0):    
            S_i=S_initial
            S_i[j/2]+=Z[i*10+j]
    S_i=np.dot(S_i.T,S_i)
    S+=S_i

S_inv=np.linalg.inv(S)
#Eigen vectors and values:
S_inv_mul_B=np.matmul(S_inv,Sb)
print("shapes of S_inv and Sb,S_inv_mul_ﻵ",S_inv,Sb,S_inv_mul_B)
"""
#commenting those because of high processing

eigenvals,eigenvecs = np.linalg.eig(S_inv_mul_B)
#sorting eigen values in descending order
sort_idx = np.argsort(eigenvals)[::-1]
eigenvals = eigenvals[sort_idx]
eigenvecs = eigenvecs[:,sort_idx]
print("shape of eigen values",eigenvals)
print("shape of eigen vectors",eigenvecs)

lamb=eigenvals[0]
print("highest eigen value")
print(lamb)
print("corresponding eigen vector (direction)")
w=eigenvecs[:,0]
print(w)
"""

