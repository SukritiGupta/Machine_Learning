import numpy as np


from scipy.spatial.distance import euclidean

datax = np.genfromtxt('train.csv', delimiter=',')

pred = np.genfromtxt('test.csv', delimiter=',')
tdig=pred[:,784]
pred=np.delete(pred,784,1)
pred=pred/255.0
tot=pred.shape[0]

def predictor(i,j):

	dig=datax[:,784]
	# m=dig.shape[0]
	print(i)
	print(j)
	d4=datax[np.ix_(dig==i,)]
	d5=datax[np.ix_(dig==j,)]
	print(d4.shape)
	d4=np.delete(d4,784,1)
	print(d4.shape)

	d5=np.delete(d5,784,1)
	d4=d4/255.0
	d5=d5/255.0
	n4=d4.shape[0]
	n5=d5.shape[0]

	TD=np.concatenate((d4,d5),axis=0)
	o=np.concatenate((np.ones((n4,1))*(-1.0),np.ones((n5,1))),axis=0)
	m=n4+n5

	ED=np.zeros((m,tot))
	for i1 in range(m):
 		for j1 in range(tot):
 			# print(1)
 			ED[i1][j1]=(euclidean(TD[i1],pred[j1]))**2
			# ED[i][j] = (euclidean(TD[i],tdata[j]))**2
			# pass
	ED=-0.05*ED
	ED=np.exp(ED)
	alpha=np.genfromtxt(("d"+str(i)+str(j)+".csv"),delimiter=',')
	alphalen=alpha.shape[0]
	alpha=alpha.reshape((alphalen,1))
	b=alpha[alphalen-1]
	alpha=np.delete(alpha,alphalen-1,0)
	aiyi=alpha*o
	aiyi=aiyi.T
	n2check=(np.dot(aiyi,ED)).T
	ret=np.zeros(tot)
	for x in range(0,tot):
		if ((n2check[x]+b)<0.0):
			ret[x]=i
		else:
			ret[x]=j

	fn="calc"+str(i)+str(j)+".csv"
	np.savetxt(fn,ret)
	return ret


store=np.zeros((tot,45))
k=0
for i in range(10):
	for j in range(i+1,10):
		store[:,k]=predictor(i,j)
		k=k+1

np.savetxt("work.csv",store)

# for i in range(m):
# 	store[m]
