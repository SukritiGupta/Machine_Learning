import pylab
import numpy as np

datax = np.genfromtxt('weightedX.csv', delimiter=',')
datay = np.genfromtxt('weightedY.csv', delimiter=',')

X=np.concatenate((np.ones((datax.size,1)), datax.reshape((datax.size,1))), axis=1)
Y=datay.T

temp=np.linalg.inv(np.dot(X.T,X))
theta=np.dot(temp, np.dot(X.T,Y))
	
theta0=theta[0]
theta1=theta[1]

pylab.plot(datax,datay,'o')
pylab.plot(datax,theta0+theta1*datax, '-k')
print(theta0, theta1)
pylab.show()
