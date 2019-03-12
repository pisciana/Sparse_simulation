#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:45:08 2019

@author: anaclaudia
"""
import numpy as np
import matplotlib.pyplot as plt


def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights)
    a = feed_forward(X, weights)
    delta = a[-1] - Y
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1):
        delta = (a[i] > 0) * delta.dot(weights[i].T)
        grads[i-1] = a[i-1].T.dot(delta)
    return grads / len(X)

def buildRandomBoolDataSet(Nsamples=100,Dinput=100,seed=0):
    np.random.seed(seed)
    features, labels = (np.random.randint(2,size=(Nsamples,Dinput)), np.eye(Nsamples))
    features = (np.float32(features)*2-1)/np.sqrt(Dinput)
    labels = np.float32(labels)
    return features, labels, features, labels
  
#%%
    
Nsamples=2000
Dinput=100
Doutput = 2000
seed=0
trX, trY, teX, teY = buildRandomBoolDataSet(Nsamples, Dinput, seed)
weights = [np.random.randn(*w) * 0.1 for w in [(Dinput, Nsamples), (Nsamples, Doutput)]]
num_epochs, batch_size, learn_rate = 30, 100, 0.1
acc = np.zeros(num_epochs) 
for i in range(num_epochs):
    for j in range(0, len(trX), batch_size):
        X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
        weights -= learn_rate * grads(X, Y, weights)
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    acc[i] = np.mean(prediction == np.argmax(teY, axis=1))
    #print(i, np.mean(prediction == np.argmax(teY, axis=1)))
    
  #%%  
#plt.scatter(acc[:0]) 

plt.plot(acc)

plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()

#%%

