'''
Created on Oct 28, 2017
'''

from theano import tensor as T
from theano.tensor.nnet import conv2d
import theano
import pylab
from PIL import Image 
import theano.tensor as T
import os
import re
import numpy as np
import pandas as pd
from scipy.misc import imresize
from scipy.misc import imread
import lasagne,theano,scipy
import sys,pickle,os,re
import theano.tensor as T
import numpy as np
from lasagne.nonlinearities import softmax,very_leaky_rectify
from lasagne.layers import InputLayer, DenseLayer, get_output,MaxPool2DLayer,Conv2DLayer,Layer
from lasagne.updates import sgd, apply_momentum,apply_nesterov_momentum,adagrad
from scipy.misc import *
from collections import OrderedDict
from lasagne.init import Constant, GlorotUniform



# pred=theano.function([x1,y1],outputs=prediction,on_unused_input='ignore')

### begin to train


rng=np.random


imagename=filter(lambda x:re.search(r".jpg",x),os.listdir("C:/Users/zjutc/Desktop/training_set/images")) # get all image names from folder
y=pd.read_csv("C:/Users/zjutc/Desktop/training_set/label_train.csv")# get labels 

df=pd.DataFrame(y)
df["imagename"]=imagename
df = df.sample(frac=1).reset_index(drop=True)  # shuffle 

trainX=[]
testX=[]

for i in df.loc[:2499,"imagename"]:
    img=imread("C:/Users/zjutc/Desktop/training_set/images"+"/"+i)
    if len(img.shape)==2:
        img_conv=np.zeros([img.shape[0],img.shape[1],3])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_conv[i,j]=img[i,j]    
        img=img_conv
    img=imresize(img,[128,128,3])
    
    trainX.append(img.transpose([2,0,1]))
    
trainY=df.iloc[:2500,1]

for i in df.loc[2500:,"imagename"]:
    img=imread("C:/Users/zjutc/Desktop/training_set/images"+"/"+i)
    if len(img.shape)==2:
        img_conv=np.zeros([img.shape[0],img.shape[1],3])
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_conv[i,j]=img[i,j]
                
        img=img_conv
    img=imresize(img,[128,128,3])
    
    testX.append(img.transpose([2,0,1]))

testY=df.iloc[2500:,1]

trainX=[x/256.0 for x in trainX]
testX=[x/256.0 for x in testX]

trainY=trainY.astype("int64")
testY=testY.astype("int64")



### start to build the CNN network
x1=T.tensor4('x1',dtype='float64')
y1=T.vector('y1',dtype='int64')
batchsize=100
network=InputLayer(shape=(None,3,128,128),input_var=x1)
network=Conv2DLayer(network,48,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
network=MaxPool2DLayer(network,(2,2))
network=Conv2DLayer(network,64,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
network=MaxPool2DLayer(network,(2,2))
network=Conv2DLayer(network,96,(5,5),nonlinearity=very_leaky_rectify,W=GlorotUniform('relu'))
network=MaxPool2DLayer(network,(3,3))
network=DenseLayer(network,512,nonlinearity=very_leaky_rectify,W=lasagne.init.GlorotNormal())

network=DenseLayer(network,2,nonlinearity=softmax)

rate=theano.shared(.0002)
params = lasagne.layers.get_all_params(network)
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction,y1)
loss = loss.mean()
updates_sgd = adagrad(loss, params, learning_rate=rate)
updates = apply_nesterov_momentum(updates_sgd, params, momentum=0.9)


train_model = theano.function([x1,y1],outputs=loss,updates=updates)

pred = theano.function([x1,y1],outputs=lasagne.objectives.categorical_crossentropy(prediction,y1))

### begin to train
renewtrain=len(trainX)/batchsize
renewtest=len(testX)/batchsize
for i in range(15000):
    if i>325 and i<3000:
        rate.set_value(.001)
    elif i>6500 and i<15000:
        rate.set_value(.0005)            
    i1=i%renewtrain
    tindex=range(i1*batchsize,(i1+1)*batchsize)
    newloss=train_model([trainX[i] for i in tindex],[trainY[i] for i in tindex])
    print 'in %d round, the loss function is %f'%(i+1,newloss) 
    if i%renewtrain==0:
        tt1=range(500)
        pred1=0.
        tmp_x=testX[tt1]
        tmp_y=testY[tt1]
        pred1+=sum(pred(tmp_x,tmp_y))
        print 'in %d circle, the total test error is %f'%(i/renewtest+1,pred1/250.0)
        tmp=rng.permutation(2500)
        trainX=trainX[tmp]
        trainY=trainY[tmp]


    