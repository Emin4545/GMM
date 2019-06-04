import numpy as np
from scipy.stats import multivariate_normal
from sklearn.datasets import load_iris
import random
all_data = load_iris()
data=all_data.data



def normal_i_k(mu,cova,vector):
    N = multivariate_normal.pdf(vector, mean=mu, cov=cova)   
    return N

def population_i_k(normal,weghit,k):
    p=0
    for i in range(k):
        p=p+(normal[i]*weghit[i])
    return p

def E(k,vector1,weight_vector,cova1,mu1,k_choose):
    gaus=[]
    for i in range(k):
        gaus.append(normal_i_k(mu1[i],cova1[i],vector1))
    tempp=population_i_k(gaus,weight_vector,k)
    r_i_k=(weight_vector[k_choose]*normal_i_k(mu1[k_choose],cova1[k_choose],vector1))/tempp
    return r_i_k

def M(r_matrix,k,X):
    new_mu=[]
    new_w=[]
    new_cova=[]
    for i in range(k):
        temp_mu=0
        temp_mu2=0
        new_w.append(r_matrix[:,i].sum()/r_matrix.sum())
        for x in range(len(X)):
            temp_mu+=r_matrix[x,i]*X[x]
            temp_mu2+=r_matrix[x,i]
        temp_mu=temp_mu/temp_mu2
        new_mu.append(temp_mu)
    for i in range(k):
        temp=np.zeros((X.shape[1],X.shape[1]))
        for x in range(len(X)):
            temp+=r_matrix[x,i]*(X[x]-new_mu[i]).reshape(4,1)@(X[x]-new_mu[i]).reshape(1,4)
        temp=temp/r_matrix[:,i].sum()
        new_cova.append(temp)
    return new_w,new_mu,new_cova
 
def main(k,X,iterations):    
    rand = list(range(len(X)))
    random.shuffle(rand)
    first_mu=[]
    first_cova=[]
    first_weight_vector=[]
    r_matrix1=np.zeros((len(X),k))
    for i in rand[:k]:
        first_mu.append(X[i])
        first_cova.append(np.eye(X.shape[1]))
        first_weight_vector.append(1/k)
    for iter in range(iterations):
        for x in range(len(r_matrix1)):
            for y in range(k):
                r_matrix1[x,y]=E(k,X[x],first_weight_vector,first_cova,first_mu,y)
        first_weight_vector,first_mu,first_cova=M(r_matrix1,k,X)
    return r_matrix1

rr=main(3,data,50)



                
                
    
    
    
        
        
    
        
    
    
    


        
        

