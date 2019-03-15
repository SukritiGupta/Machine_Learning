import numpy as np
import math
from svmutil import *
import time

datax = np.genfromtxt('train.csv', delimiter=',')

s=time.time()
dig=datax[:,784]
# m=dig.shape[0]
d4=datax[np.ix_(dig==4,)]
d5=datax[np.ix_(dig==5,)]
# print(d4.shape)
d4=np.delete(d4,784,1)
d5=np.delete(d5,784,1)
d4=d4/255.0
d5=d5/255.0
X=np.concatenate((d4,d5), axis=0)
n4=d4.shape[0]
n5=d5.shape[0]
Y=np.concatenate((np.ones(n4)*(-1),np.ones(n5)),axis=0)
prob=svm_problem(Y,X)

s=time.time()
m=svm_train(prob,'-g 0.05 -t 2 -c 1 -b 1') #for gaussian kernel
#m=svm_train(prob,'-t 0 -c 1 -b 1') for linear kernel
end=time.time()
print(end-s)
datax = np.genfromtxt('test.csv', delimiter=',')

tdig=datax[:,784]
# m=dig.shape[0]
td4=datax[np.ix_(tdig==4,)]
td5=datax[np.ix_(tdig==5,)]
# print(d4.shape)
td4=np.delete(td4,784,1)
td5=np.delete(td5,784,1)
td4=td4/255.0
td5=td5/255.0
tX=np.concatenate((td4,td5), axis=0)
tn4=td4.shape[0]
tn5=td5.shape[0]
tY=np.concatenate((np.ones(tn4)*(-1),np.ones(tn5)),axis=0)

p_label, p_acc, p_val = svm_predict(tY, tX, m)
print(p_acc)

# *.*
# optimization finished, #iter = 2079
# nu = 0.105364
# obj = -169.284295, rho = -0.178495
# nSV = 1299, nBSV = 8
# Total nSV = 1299
# .*.*
# optimization finished, #iter = 2068
# nu = 0.105813
# obj = -170.256012, rho = -0.193045
# nSV = 1300, nBSV = 12
# Total nSV = 1300
# .*.*
# optimization finished, #iter = 2055
# nu = 0.105919
# obj = -170.671818, rho = -0.185858
# nSV = 1301, nBSV = 15
# Total nSV = 1301
# .*.*
# optimization finished, #iter = 2072
# nu = 0.105215
# obj = -169.215875, rho = -0.167640
# nSV = 1317, nBSV = 9
# Total nSV = 1317
# .*.*
# optimization finished, #iter = 2044
# nu = 0.105219
# obj = -169.120107, rho = -0.189593
# nSV = 1291, nBSV = 8
# Total nSV = 1291
# .*.*
# optimization finished, #iter = 2317
# nu = 0.092016
# obj = -185.120537, rho = -0.185630
# nSV = 1459, nBSV = 12
# Total nSV = 1459
# 26.875377416610718
# Model supports probability estimates, but disabled in predicton.
# Accuracy = 99.8933% (1872/1874) (classification)
# (99.89327641408752, 0.004268943436499467, 0.9957257647146011)
