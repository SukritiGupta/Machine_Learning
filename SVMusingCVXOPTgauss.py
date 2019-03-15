# import pylab
import numpy as np
import math
import time
from cvxopt.solvers import qp
from cvxopt import matrix
from scipy.spatial.distance import squareform, pdist
import time

datax = np.genfromtxt('train.csv', delimiter=',')

s=time.time()
dig=datax[:,784]
# m=dig.shape[0]
d4=datax[np.ix_(dig==4,)]
d5=datax[np.ix_(dig==5,)]
print(d4.shape)
d4=np.delete(d4,784,1)
print(d4.shape)

d5=np.delete(d5,784,1)
d4=d4/255.0
d5=d5/255.0
n4=d4.shape[0]
n5=d5.shape[0]

TD=np.concatenate((d4,d5),axis=0)
temp=squareform(pdist(TD,'euclidean'))
K=np.exp(-0.05*(temp**2))
s=K.shape[0]
o=np.concatenate((np.ones((n4,1))*(-1.0),np.ones((n5,1))),axis=0)
YY=np.dot(o,o.T)
P=K*YY



d4=d4*(-1)

X=np.concatenate((d4, d5), axis=0)
m=X.shape[0]

q=np.ones((m,1))*(-1)
ones=np.ones((m,1))
G=np.concatenate((np.diag(np.ones(m))*(1.0),np.diag(np.ones(m)*(-1.0))),axis=0)
h=np.concatenate((ones*1.0,np.zeros((m,1))),axis=0)
A=np.concatenate((np.ones((n4,1))*(-1.0),np.ones((n5,1))*1.0),axis=0).T
b=np.array([0]).reshape(1,1)
# P=P.astype(float) 
# q=q.astype(float) 
# G=G.astype(float) 
# h=h.astype(float) 
# A=A.astype(float) 
# b=b.astype(float) 
print(P.shape)
print(q.shape)
print(G.shape)
print(h.shape)
print(A.shape)
print(b.shape)

P=matrix(P)
print(P.typecode)
q=matrix(q, tc='d')
G=matrix(G, tc='d')
h=matrix(h, tc='d')
A=matrix(A, tc='d')
b=matrix(b, tc='d')
print(q.typecode)
print(G.typecode)
print(h.typecode)
print(P.typecode)
print(b.typecode)

alpha=qp(P,q,G,h,A,b)
print(alpha)
end=time.time()
print(end-s)
# print(alpha['x'])
alpha=alpha['x']

alpha=np.array(alpha)
print(alpha.shape)
# print(alpha)
sv=TD[np.ix_(alpha[:,0]>0.000001,)]#float return hota hai
# print(sv)
print(sv.shape)

from scipy.spatial.distance import euclidean
def gsum(z):
    ret=0
    for i in range(m):
        if(alpha[i]>0.000001):
            ret+=alpha[i]*o[i][0]*(math.exp(-0.05*((euclidean(TD[i],z))**2)))
    return ret    

loopp=n4+n5
for i in range(loopp):
    if(alpha[i]>0.00001):
        if(i<n4):
            b=-1-gsum(TD[i])
            break
        else:
            b=1-gsum(TD[i])
            # print("hiyaaa")
            break
            
print(b)


pred = np.genfromtxt('test.csv', delimiter=',')
tdig=pred[:,784]
tdata4=pred[np.ix_(tdig==4,)]
tdata5=pred[np.ix_(tdig==5,)]

tdata4=np.delete(tdata4,784,1)
tdata5=np.delete(tdata5,784,1)
tdata4=tdata4/255.0
tdata5=tdata5/255.0
# b=b*255

t4=tdata4.shape[0]
t5=tdata5.shape[0]
tot=t4+t5


stf=time.time()
correct=0
l=[]
for i in range(t4):
    if ((gsum(tdata4[i])+b)<0.0):
        correct=correct+1
for i in range(t5):
    if (( gsum(tdata5[i])+b)>0.0):
        correct=correct+1

    
print(correct)
print(tot)
print(correct*1.0/tot)
etf=time.time()
print(etf-stf)
