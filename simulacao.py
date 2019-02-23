#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from modelo import Model

def buildRandomBoolDataSet(Nsamples=2000,Dinput=1000,seed=0):
    np.random.seed(seed)
    features, labels = (np.random.randint(2,size=(Nsamples,Dinput)), np.eye(Nsamples))
    features = (np.float32(features)*2-1)/np.sqrt(Dinput)
    labels = np.float32(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    return dataset
  
sess = tf.InteractiveSession()

def simulacao( input_dim, num_samples, sparseness, learning_rate, model_seed,data_seed, nsteps = 1000, max_count = 100 ):    

    acc = np.zeros(nsteps+1)
    myDataSet = buildRandomBoolDataSet(num_samples,input_dim,seed=data_seed).batch(num_samples)
    
    myIter = myDataSet.make_initializable_iterator()
    myFeatures, myLabels = myIter.get_next()               
    x = myFeatures
    y_ = myLabels              
    model = Model(x, y_, sparseness, learning_rate, seed=model_seed)
    model.set_vanilla_loss()              
    # initialize variables
    sess.run(tf.global_variables_initializer())    
    sess.run(myIter.initializer)
    acc[0] = model.accuracy.eval()            
    count  = 1            
    for ss in np.arange(1,nsteps+1):
        sess.run(myIter.initializer)
        model.reduce_sparsness()
        model.train_step.run()
        model.increase_sparsness()
        sess.run(myIter.initializer)
        acc[ss] = model.accuracy.eval()
        if(acc[ss] == acc[ss-count]):
            count += 1
        else:
            count = 1
        if(count==max_count):
            acc[ss:] = acc[ss]
            break

    return acc
