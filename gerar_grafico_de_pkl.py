#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:49:20 2019

@author: anaclaudia
"""

import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt

dirr ='DIRETORIO_DOS_DADOS'

input_dim = 100
learning_rate = 0.1
model_seed = 0
data_seed = 0
nsteps = 1000
max_count = 100
sparse = np.array([0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
num_samplesV = np.power(10,np.arange(0.2,4.3,0.2))
data_seedV = np.arange(20)
cont = 0
acc = np.ones((len(sparse),len(num_samplesV),len(data_seedV))) * np.nan

for zz,sparseness in enumerate(sparse):
    for ii,num_samples in enumerate(num_samplesV):
        for jj,data_seed in enumerate(data_seedV):
            try:           
            
                fffile = dirr + '/ana_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d.pkl' % (int(input_dim),int(num_samples),int(sparseness*100),int(learning_rate*100),int(model_seed),int(data_seed),int(nsteps),int(max_count)) 
        
                fp = gzip.open(fffile,'rb')
                aa = pickle.load(fp)
                acc[zz,ii,jj] = aa[0]    
                cont += 1
                print(cont)
            except:
                print(dirr + '/ana_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d.pkl **' % (int(input_dim),int(num_samples),int(sparseness*100),int(learning_rate*100),int(model_seed),int(data_seed),int(nsteps),int(max_count)) )
                pass



# %%

ax = plt.subplot(111)
for zz,sparseness in enumerate(sparse):  
    plt.semilogx(num_samplesV,np.median(acc[zz,:,:],axis=1), label="spars=%1.2f"%(sparseness,))

plt.xlabel('NSamples ')
plt.ylabel('Accuracy')
plt.grid(True)
#plt.axis([5,9, 0.0,1.0])
leg = plt.legend(bbox_to_anchor=(1.1, 1.05))
leg.get_frame().set_alpha(0.5)
plt.show()

plt.savefig('accuracy.eps')