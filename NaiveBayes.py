# build vocab
# train: calculate the value of (phi_k| y) *5

# done in python 2
# find max p(y|x) for all 5 classes //ignore words that were not part of training set
import json
import string
import math
import time
# from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer


l=[0,0,0,0,0]
vocab={}
n=[0,0,0,0,0]

start=time.time()
with open('train.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        s=s-1
        # t=word_tokenize(tweet['text'])#counts fullstops and stuff as well
        # t=tweet['text'].split()
        t=CountVectorizer().build_analyzer()(tweet['text'])
        l[s]=l[s]+len(t)
        n[s]=n[s]+1
        for w in t:
            if w in vocab:
                vocab[w][s]=vocab[w][s]+1
            else:
                wl=[1,1,1,1,1]
                wl[s]=wl[s]+1
                vocab[w]=wl
unchangvocab={}
unchangvocab=vocab

print(vocab['perfect'])
v=len(vocab)
print(len(vocab))
print(l)
ld=[0,0,0,0,0]
size=0
for i in [0,1,2,3,4]:
	l[i]+=v
	ld[i]=math.log(l[i])
	print(i)
	size+=n[i]
for i in [0,1,2,3,4]:
	print(i)
	print(n[i]/size)
	n[i]=math.log(n[i]/size)
print(l)
print(n)
print(vocab['perfect'])
logvocab={}
for w,x in vocab.items():
    for i in [0,1,2,3,4]:
#         print(x[i])
#         print(w)
        x[i]=math.log(x[i])-ld[i]
    logvocab[w]=x

correct=0
total=0
# n=[-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
print(n)
p=[0,0,0,0,0]
# sleep(100)
with open('test.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        # t=tweet['text'].split()
        t=CountVectorizer().build_analyzer()(tweet['text'])
        
        for i in [0,1,2,3,4]:
            p[i]=n[i]       
        
        # print(n)
        for w in t:
            if w in logvocab:
                x=logvocab[w]
                for i in [0,1,2,3,4]:
                    p[i]+=x[i]
        prediction=p.index(max(p))+1
        # print(p)
        if s==prediction:
            correct+=1
        total+=1
        # if prediction!=5:
        #     print(prediction)

print("test")

print(correct)
print(total)
print(correct/total*100)

end=time.time()
# print(end-start)

correct=0
total=0
# n=[-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
print(n)
# sleep(100)
with open('train.json', 'r') as f:

    for line in f:
        tweet = json.loads(line)
        s=(int)(tweet['stars'])
        # t=tweet['text'].split()
        t=CountVectorizer().build_analyzer()(tweet['text'])
        
        for i in [0,1,2,3,4]:
            p[i]=n[i]       
        
        # print(n)
        for w in t:
            if w in logvocab:
                x=logvocab[w]
                for i in [0,1,2,3,4]:
                    p[i]+=x[i]
        prediction=p.index(max(p))+1
        # print(p)
        if s==prediction:
            correct+=1
        total+=1
        # if prediction!=5:
        #     print(prediction)

print("train")

print(correct)
print(total)
print(correct/total*100)

end=time.time()


# [712, 657, 1836, 8104, 18215]
# 186775
# [10843518, 5824327, 7304887, 12777019, 19583362]
# 0
# 1
# 2
# 3
# 4
# 0
# 0.15011255029240642
# 1
# 0.08140826216365785
# 2
# 0.10977018800759808
# 3
# 0.21985446985446985
# 4
# 0.43885452968186783
# [11030293, 6011102, 7491662, 12963794, 19770137]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# [712, 657, 1836, 8104, 18215]
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# test
# 80278
# 133718
# 60.03529816479457
# [-1.8963699309639557, -2.508278510342461, -2.20936629854321, -1.5147894521793739, -0.8235872882885679]
# train
# 337262
# 534872
# 63.05471215543158


