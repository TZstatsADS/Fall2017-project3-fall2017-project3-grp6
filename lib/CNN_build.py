'''
Created on Oct 28, 2017

@author: Pinren, Xiaoyu
'''
import cPickle as pickle
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
from scipy.misc import imshow
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

from lasagne.nonlinearities import leaky_rectify, softmax

# pred=theano.function([x1,y1],outputs=prediction,on_unused_input='ignore')

if __name__ == "__main__":
	
    ### load image data
    try:
        dir_path=sys.argv[1] # the main folder

    except IndexError:
        print "invalid input!"
        print "it should be CNN_build.py [dir_path] )"
        sys.exit()
	rng=np.random
    
    
    imagename=filter(lambda x:re.search(r".jpg",x),os.listdir(os.path.join(dir_path,"training_set/images"))) # get all image names from folder
    imagename.sort(key=lambda f: int(filter(str.isdigit, f)))
    y=pd.read_csv(os.path.join(dir_path,"training_set/label_train.csv"))# get labels
    
    df=pd.DataFrame(y)
    df["imagename"]=imagename
    #df = df.sample(frac=1).reset_index(drop=True)  # shuffle 
    
    trainX=[]
    testX=[]
    
    # get trainX from all photos
    trainY=df.iloc[:2500,1]
    for i in df.loc[:2499,"imagename"]:
        img=imread(dir_path+"/training_set/images"+"/"+i)
        if len(img.shape)==2:
            img_conv=np.zeros([img.shape[0],img.shape[1],3])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_conv[i,j]=img[i,j]    
            img=img_conv
        img=imresize(img,[64,64,3])    
        trainX.append(img.transpose([2,0,1]))
        
    
    
    # get testX from all photos
    for i in df.loc[2500:,"imagename"]:
        img=imread(dir_path+"/training_set/images"+"/"+i)
        if len(img.shape)==2:
            img_conv=np.zeros([img.shape[0],img.shape[1],3])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_conv[i,j]=img[i,j]
                    
            img=img_conv
        img=imresize(img,[64,64,3])
        
        testX.append(img.transpose([2,0,1]))
    
    testY=df.iloc[2500:,1]
    
    trainX=[x/256.0 for x in trainX]
    testX=[x/256.0 for x in testX]
    
    trainY=trainY.astype("int64")
    testY=testY.astype("int64")
    
    trainX=np.asarray(trainX)
    trainY=np.asarray(trainY)
    testX=np.asarray(testX)
    testY=np.asarray(testY)
    
    ### start to build the CNN network
    
    input_var = T.tensor4('X',dtype='float64')
    target_var = T.vector('y',dtype='int64')
    batchsize=20
    network = lasagne.layers.InputLayer((None, 3, 64, 64), input_var)
    network = lasagne.layers.Conv2DLayer(network, 32, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
                                         
    network = lasagne.layers.Conv2DLayer(network, 64, (3, 3),
                                         nonlinearity=leaky_rectify)
    
                                         
                                         
    network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
    
    network = lasagne.layers.Conv2DLayer(network, 96, (3, 3),
                                         nonlinearity=leaky_rectify)
    network = lasagne.layers.Pool2DLayer(network, (3, 3), stride=2, mode='max')
    
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                        128, nonlinearity=leaky_rectify,
                                        W=lasagne.init.Orthogonal())
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, 0.5),
                                        3, nonlinearity=softmax)    
    
    # create loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(
            network, lasagne.regularization.l2)
    
    # create parameter update expressions
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.rmsprop(loss, params, learning_rate=0.001)
    
    # compile training function that updates parameters and returns training loss
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    
    
    ### begin to train
    # train network (assuming you've got some training data in numpy arrays)
    train_times=len(trainX)/batchsize
    
    
    print("Training ...")
    
    for epoch in range(100):
        loss = 0  
        for i in range(train_times):  
            
            input_batch=trainX[i*batchsize:(i+1)*batchsize]
            target_batch=trainY[i*batchsize:(i+1)*batchsize]
    
    
            loss += train_fn(input_batch, target_batch)
    
    
        
        Net_FileName = 'training_set/CNNmodel.pkl'
        netInfo = {'network': network, 'params': lasagne.layers.get_all_param_values(network)}
        pickle.dump(netInfo, open(os.path.join(dir_path, Net_FileName), 'wb'),protocol=pickle.HIGHEST_PROTOCOL) # save models
        
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    
        
        prediction=predict_fn(testX)
        prediction=np.asarray(prediction)
        error=sum(prediction!=testY)/500.0
        print "Epoch %d: Loss %g" % (epoch + 1, loss ) # report loss
        print "Epoch %d: Prediction Error %g" %(epoch+1,error) # report prediction error
        
    
    
    # when epoch is 69, the test accuracy is 0.9
    