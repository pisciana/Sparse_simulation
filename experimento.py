#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 21:31:17 2019

@author: anaclaudia
"""

import tensorflow as tf
import numpy as np  
import matplotlib.pyplot as plt
from modelo import Model


  
def buildRandomBoolDataSet(Nsamples=2000,Dinput=1000,seed=0):
    np.random.seed(seed)
    features, labels = (np.random.randint(2,size=(Nsamples,Dinput)), np.eye(Nsamples))
    features = (np.float32(features)*2-1)/np.sqrt(Dinput)
#    labels = (np.float32(features)*2-1)/np.sqrt(Doutput)
    labels = np.float32(labels)
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    return dataset


sess = tf.InteractiveSession()
input_dim = 100
learning_rate = 0.5
model_seed = 0
data_seed = 0
nsteps = 2
max_count = 100
num_samplesV = 1000
sparse = 0.1


data_seedV = np.arange(3)

myDataSet = buildRandomBoolDataSet(num_samplesV,input_dim, data_seed).batch(num_samplesV)                
myIter = myDataSet.make_initializable_iterator()
myFeatures, myLabels = myIter.get_next()               
x = myFeatures
y_ = myLabels        
    


model = Model(x, y_, sparse, learning_rate, seed=data_seed) # simple 2-layer network
model.set_vanilla_loss()        

sess.run(tf.global_variables_initializer())   
sess.run(myIter.initializer)

count  = 1       
for ss in np.arange(1,nsteps+1):
    sess.run(myIter.initializer)
    #model.reduce_sparsness()
    model.train_step.run()    
    
    
    sess.run(myIter.initializer)
    acc = model.accuracy.eval() 
    
    sess.run(myIter.initializer)    
    ce = model.cross_entropy.eval()  

    sess.run(myIter.initializer)    
    w1 = model.W1.eval()  


    print("acuracia: {} / Entropia: {}".format(acc,ce))
    print("W1:{}".format(w1))
    print("***************ANTES********************")
    #sess.run(myIter.initializer)
    #y = model.y.eval()  
    

    #np.savetxt('filesemreduz{}.txt'.format(ss), y, delimiter=",")
    #arquivo.write(y)#
    #   arquivo.close()
    
    #acc[ss] = model.accuracy.eval()
sess.close()

# %%
for zz,sparseness in enumerate(sparse):
    for ii,num_samples in enumerate(num_samplesV):
        for jj,data_seed in enumerate(data_seedV):
                
                myDataSet = buildRandomBoolDataSet(num_samples.astype(np.int64),input_dim, data_seed).batch(num_samples)                
                myIter = myDataSet.make_initializable_iterator()
                myFeatures, myLabels = myIter.get_next()               
                x = myFeatures
                y_ = myLabels              
                #model = Model(x, y_, spars, 0.1) # simple 2-layer network
                #print("data_seed")
                #print(data_seed)

                model = Model(x, y_, sparseness, learning_rate, seed=data_seed) # simple 2-layer network
                model.set_vanilla_loss()              
                # initialize variables
                sess.run(tf.global_variables_initializer())    
                sess.run(myIter.initializer)
                #acc[0] = model.accuracy.eval()            
                count  = 1            
                for ss in np.arange(1,nsteps+1):
                    sess.run(myIter.initializer)
                    model.reduce_sparsness()
                    model.train_step.run()
                    #model.aumenta spar
                    sess.run(myIter.initializer)
                    #acc[ss] = model.accuracy.eval()
                    
                    acc[zz,ii,jj] = model.accuracy.eval()
    
                    
sess.close()
 
# %%

ax = plt.subplot(111)
#plt.plot(acc[:,:,0])
for zz,sparseness in enumerate(sparse):  
    plt.plot(num_samplesV,np.median(acc[zz,:,:],axis=1), label="spars=%1.2f"%(sparseness,))
#    plt.semilogx(num_samplesV,np.percentile(acc[zz,:,:],25,axis=1))
#    plt.semilogx(num_samplesV,np.percentile(acc[zz,:,:],75,axis=1))

plt.xlabel('NSamples ')
plt.ylabel('Accuracy')
plt.grid(True)
#plt.axis([5,9, 0.0,1.0])
leg = plt.legend(bbox_to_anchor=(1.1, 1.05))
leg.get_frame().set_alpha(0.5)
plt.show()

plt.savefig('accuracy.eps')
   