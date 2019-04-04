import numpy as np
import math
from anytree import AnyNode, RenderTree

def normalise(data, med):
	data=np.delete(data,0,1)

	# X1
	data[:,0]=(data[:,0]>med[0]).astype(int) 
	# X2
	data[:,1]=(data[:,1]>1).astype(int) 
	# x3, x4: categorical
	# x5
	data[:,4]=(data[:,4]>med[4]).astype(int)
	# x6-x11: categorical
	for i in range(5,11):
		data[:,i]=data[:,i]+2

	# x12
	data[:,11]=(data[:,11]>med[11]).astype(int)
	data[:,12]=(data[:,12]>med[12]).astype(int)
	data[:,13]=(data[:,13]>med[13]).astype(int)
	data[:,14]=(data[:,14]>med[14]).astype(int)
	data[:,15]=(data[:,15]>med[15]).astype(int)
	data[:,16]=(data[:,16]>med[16]).astype(int)
	data[:,17]=(data[:,17]>med[17]).astype(int)
	data[:,18]=(data[:,18]>med[18]).astype(int)
	data[:,19]=(data[:,19]>med[19]).astype(int)
	data[:,20]=(data[:,20]>med[20]).astype(int)
	data[:,21]=(data[:,21]>med[21]).astype(int)
	data[:,22]=(data[:,22]>med[22]).astype(int)
	return data

def calculate_entropy(d):
	# print("inside calculate_entropy")
	# print(d)
	d0=d[np.ix_(d[:,23]==0,)].shape[0]
	d1=d[np.ix_(d[:,23]==1,)].shape[0]
	tot=d0+d1
	if (tot==0 or d0==0 or d1==0):
		return 0
	p0=d0/tot
	p1=d1/tot
	ret=0
	ret+=p0*math.log(p0)
	ret+=p1*math.log(p1)
	return (-1.0*ret)

def find_entropy_on_attribute(data, att):
	noc=2
	if (att==2):
		noc=7
	if (att==3):
		noc=4
	if (att>=5 and att<=10):
		noc=12
	tot=0
	n=[None]*noc
	p=[None]*noc
	ent=[None]*noc
	for i in range(noc):
		cdat=data[np.ix_(data[:,att]==i,)]
		ent[i]=calculate_entropy(cdat)
		n[i]=cdat.shape[0]
		tot+=n[i]
	if tot==0:
		print(data, att,noc)
		exit()
	ret=0
	for i in range(noc):
		p[i]=n[i]/tot
		ret+=p[i]*ent[i]
	return ret 

def split(data, attr_list, node):

	t=np.sum(data[:,23])
	dsize=data.shape[0]

	if attr_list==[False]*23:
		if t>(dsize/2.0):
			node.attn=1001
		else:
			node.attn=1000
		return
	if (t==0):
		node.attn=1000
		return
	if (t==dsize):
		node.attn=1001
		return

	l=attr_list.copy()
	min_e=100
	spl_att=100
	for i in range(23):
		if (l[i]==True):
			temp=find_entropy_on_attribute(data,i)
			if min_e>temp:
				min_e=temp
				spl_att=i
	noc=2
	if (spl_att==2):
		noc=7
	if (spl_att==3):
		noc=4
	if (spl_att>=5 and spl_att<=10):
		noc=12

	node.attn=spl_att
	cnode=[None]*noc

	for i in range(noc):
		cnode[i]=AnyNode(attn=100,parent=node)

	l[spl_att]=False
	for i in range(noc):
		dat=data[np.ix_(data[:,spl_att]==i,)]
		split(dat, l, cnode[i])


def predict(data, node):
	if data.size==0:
		return 0
	ret=0;
	dsize=data.shape[0]
	t=np.sum(data[:,23])
	if node.attn==1000:
		return (dsize-t)
	if node.attn==1001:
		return t

	att=node.attn
	noc=2
	if (att==2):
		noc=7
	if (att==3):
		noc=4
	if (att>=5 and att<=10):
		noc=12

	for i in range(noc):
		cdat=data[np.ix_(data[:,att]==i,)]
		# print(node)
		# print(node.children[i])
		ret+=predict(cdat, node.children[i])

	return ret


path_to_csv = "credit-cards.train.csv"
data = np.genfromtxt(path_to_csv, dtype=int, delimiter=',', skip_header=2)#***** int float

# data=data[5998:]
n=data.shape[0]
# print(data)
med=np.zeros(25)
med=np.median(data, axis=0)
med=med[1:]

data=normalise(data,med)

# print(data)

root=AnyNode(attn=100, parent=None)
l=[True]*23
split(data,l,root)

# print(RenderTree(root))


path_to_test_csv="credit-cards.test.csv"
tdata = np.genfromtxt(path_to_test_csv, dtype=int, delimiter=',', skip_header=2)#***** int float
tdata=normalise(tdata,med)
tn=tdata.shape[0]
correct=predict(tdata,root)

print(correct)
print(tn)
print(correct/tn)

path_to_test_csv="credit-cards.train.csv"
tdata = np.genfromtxt(path_to_test_csv, dtype=int, delimiter=',', skip_header=2)#***** int float
tdata=normalise(tdata,med)
tn=tdata.shape[0]
correct=predict(tdata,root)

print(correct)
print(tn)
print(correct/tn)
