import pandas
import csv
import pylab
import numpy as np

x = np.genfromtxt('linearX.csv', delimiter=',')
y = np.genfromtxt('linearY.csv', delimiter=',')
	
m=100
theta0=0
theta1=0
lr=0.001/m

while True:

	ngrad0=(np.sum(y-(theta0 + theta1*x)))
	ngrad1=(np.sum(np.multiply(y-(theta0 + theta1*x), x)))

	theta0=theta0+lr*ngrad0
	theta1=theta1+lr*ngrad1
	# print(ngrad0, ngrad1)

	if ngrad0<0.00000001 and ngrad1<0.00000001:
		break

pylab.plot(x,y,'o')
pylab.plot(x,theta0+theta1*x, '-k')
print(theta0, theta1)
pylab.show()
