import pylab
import numpy as np
import math

datax = np.genfromtxt('weightedX.csv', delimiter=',')
datay = np.genfromtxt('weightedY.csv', delimiter=',')

left=min(datax)
right=max(datax)
n=100
diff=(right-left)/n
plotx=[(left+i*diff) for i in range(n+1)]
ploty=[]

X=np.concatenate((np.ones((datax.size,1)), datax.reshape((datax.size,1))), axis=1)
Y=datay.T
m=datax.size

for x in plotx:
	W=np.diag([math.exp(((x-datax[i])**2)*(-1/1.28)) for i in range(m)])
	temp=np.linalg.inv(np.dot(X.T,np.dot(W,X)))
	theta=np.dot(temp, np.dot(X.T,np.dot(W,Y)))
	ploty.append([theta[0]+theta[1]*x])

pylab.plot(datax,datay,'o')
pylab.plot(plotx,ploty, '-k')
pylab.show()
