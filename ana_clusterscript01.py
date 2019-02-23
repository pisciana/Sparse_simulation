#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from simulacao import simulacao
import sys 
import argparse
import gzip
import pickle

def main(argv):

    parser = argparse.ArgumentParser(description='Will run a simulation instance.')
    
    parser.add_argument('data_path', metavar='data_path', nargs=1,
                   help='data_path')
    
    parser.add_argument('input_dim', metavar='input_dim', type=int, nargs=1,
                   help='input_dim')
    
    parser.add_argument('num_samples', metavar='num_samples', type=int, nargs=1,
                   help='num_samples')
    
    parser.add_argument('sparseness', metavar='sparseness', type=float, nargs=1,
                   help='sparseness')
    
    parser.add_argument('learning_rate', metavar='learning_rate', type=float, nargs=1,
                   help='learning_rate')
    
    parser.add_argument('model_seed', metavar='model_seed', type=int, nargs=1,
                   help='model_seed')    
    
    parser.add_argument('data_seed', metavar='data_seed', type=int, nargs=1,
                   help='data_seed')        

    parser.add_argument('nsteps', metavar='nsteps', type=int, nargs=1,
                   help='nsteps')
    
    parser.add_argument('max_count', metavar='max_count', type=int, nargs=1,
                   help='max_count')


    args = parser.parse_args() 
    
    data_path = args.data_path[0]
    input_dim = int(args.input_dim[0])
    num_samples = int(args.num_samples[0])
    sparseness = float(args.sparseness[0])
    learning_rate = float(args.learning_rate[0])
    model_seed = int(args.model_seed[0])
    data_seed = int(args.data_seed[0])
    nsteps = int(args.nsteps[0])
    max_count = int(args.max_count[0])

    
    print('input dimension: %d' % (input_dim))
    print('number of samples: %d' % (num_samples))
    print('sparseness: %f' % (sparseness))
    print('learning_rate: %f' % (learning_rate))
    print('model_seed: %d' % (model_seed))
    print('data_seed: %d' % (data_seed))
    print('nsteps: %d' % (nsteps))
    print('max_count: %d' % (max_count))
    
    

    fffile = data_path + '/ana_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d_%.5d.pkl' % (int(input_dim),int(num_samples),int(sparseness*100),int(learning_rate*100),int(model_seed),int(data_seed),int(nsteps),int(max_count)) 
    print('will save at: %s\n'%(fffile))


    from pathlib import Path

    my_file = Path(fffile)
    if my_file.is_file():
        
        print('file already exists... done\n')
        
    else:

        aaa = simulacao(input_dim = int(input_dim),
                    num_samples = int(num_samples),
                    sparseness = float(sparseness),
                    learning_rate = float(learning_rate),
                    model_seed = int(model_seed),
                    data_seed = int(data_seed),
                    nsteps = int(nsteps),
                    max_count = int(max_count))
    
        print(aaa[-1])
        
        accuracy = aaa[-1]
    
        with gzip.open(fffile, 'wb') as ff:
            pickle.dump([accuracy] , ff)
    

    
if __name__ == "__main__":
   main(sys.argv[1:])
   
        
