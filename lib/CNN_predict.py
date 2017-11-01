'''
Created on Oct 28, 2017

@author: Pinren
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


if __name__=="__main__":
    try:
        dir_path=sys.argv[1] # set dir_path for test folder e.g: "C:/Users/zjutc/Desktop"
        model_path=sys.argv[2] # set path of CNN model e.g."C:/Users/zjutc/Desktop/training_set/CNNmodel.pkl"
    except IndexError:
        print "invalid input!"
        print "it should be CNN_predict.py [dir_path]"
        sys.exit()

    ### load test file:
    imagename=filter(lambda x:re.search(r".jpg",x),os.listdir(os.path.join(dir_path,"test_set/images"))) # get all image names from folder
    imagename.sort(key=lambda f: int(filter(str.isdigit, f)))
    testX=[]
    # get testX from all photos
    for i in imagename:
        img=imread(dir_path+"/test_set/images"+"/"+i)
        if len(img.shape)==2:
            img_conv=np.zeros([img.shape[0],img.shape[1],3])
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img_conv[i,j]=img[i,j]
                    
            img=img_conv
        img=imresize(img,[64,64,3])
        
        testX.append(img.transpose([2,0,1]))
    
    testX=np.asarray(testX)
    
    ### setup our CNN network
    
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
    
                                        
    ### load our model:
    
    net = pickle.load(open(model_path,'rb'))
    all_params = net['params']
    
    lasagne.layers.set_all_param_values(network, all_params)
    
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    
    print "predict..."
    prediction=predict_fn(testX)
    
    
 
    
    
    
    # save prediction:
    prediction=pd.DataFrame(prediction)
    prediction.columns=['label']
    prediction.to_csv(os.path.join(dir_path,"label_CNN.csv"))