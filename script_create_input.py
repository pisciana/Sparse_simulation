#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 00:08:06 2019

@author: anaclaudia
"""
import numpy as np

base = 0
numfiles = 10

fff = []
for ii in np.arange(numfiles):
    fff.append(open('input/npad_input_file_%d.txt' % (base+ii),'w'))

pythoncode = "COMANDO_EXECUCAO_PYTHON"
datapos = " DIRETORIO_SAIDA"
scriptfile = "CODIGO_PYTHON_A_SER_EXECUTADO"

input_dim = 100
num_samples = 100000
sparseness = 0.0
learning_rate = 0.1
model_seed = 0
data_seed = 0
nsteps = 1000
max_count = 100

#num_samplesV = np.arange(100,10000,100)

#num_samplesV = np.logspace(0,4,31)[2:]
num_samplesV = np.power(10,np.arange(0.2,4.3,0.2))

data_seedV = np.arange(20)
sparsenessV = np.array([0.4,0.5,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]) #np.arange(0.0,1.0,0.1)

for jj,data_seed in enumerate(data_seedV):
    for ii,num_samples in enumerate(num_samplesV):
        for ss,sparseness in enumerate(sparsenessV):
    


            fffile = "%s %s   %s  %d %d %f %f %d %d %d %d " % (pythoncode,
                                                                scriptfile,
                                                                datapos,
                                                                input_dim,
                                                                num_samples,
                                                                sparseness,
                                                                learning_rate,
                                                                model_seed,
                                                                data_seed,
                                                                nsteps,
                                                                max_count)  
                                                                                       
            lllog = "> ./log/out_ana_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d.txt" % (input_dim,
                                                                                     num_samples,
                                                                                     sparseness*100, 
                                                                                     learning_rate*100, 
                                                                                     model_seed, 
                                                                                     data_seed, 
                                                                                     nsteps, 
                                                                                     max_count)
            
            fff[ss%numfiles].write(fffile+lllog+'\n')


for ii in np.arange(numfiles):
    fff[ii].close()
