import tensorflow as tf
import numpy as np

def weight_variable(shape,seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_, sparseness = 0.0, learning_rate = 0.5, sparse_fact = 1.0, seed = 0):

        in_dim = int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1]) 

        self.x = x 
        self.learning_rate = learning_rate
        self.sparse_fact = sparse_fact 
        self.sparseness = sparseness
        
        tf.set_random_seed(seed)
        np.random.seed(seed)

        W1 = weight_variable([in_dim,out_dim],seed)
        b1 = bias_variable([out_dim])
        
        self.mask = np.ones(W1.get_shape()).astype(np.float32)
        if(sparseness>0.0):
            self.costmask = (np.random.random(W1.get_shape())<sparseness).astype(np.float32)
            self.costmask /= np.sum(self.costmask)
            self.mask[self.costmask>0] = 0.0
        
        self.y =  tf.matmul(x,tf.multiply(W1,self.mask)) + b1 
        self.var_list = [W1, b1]

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=self.y))
        if(sparseness>0.0):
            self.cross_entropy += sparse_fact * tf.reduce_sum(tf.multiply(self.costmask.astype(np.float32),tf.square(self.var_list[0])))
        
        self.set_vanilla_loss()

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []
        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def set_vanilla_loss(self):
        out_dim = int(self.y.get_shape()[1])
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate*out_dim*self.learning_rate).minimize(self.cross_entropy)
        
    def reduce_sparsness(self):
        out_dim = int(self.y.get_shape()[1])
        if(self.sparseness > 0.0):
            self.cross_entropy += self.sparseness * self.sparse_fact * tf.reduce_sum(tf.multiply(self.costmask.astype(np.float32),tf.square(self.var_list[0])))
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate*out_dim*self.learning_rate).minimize(self.cross_entropy)
            print('Reduce Sparseness')
            print(self.cross_entropy )


    def increase_sparsness(self):
        out_dim = int(self.y.get_shape()[1])
        if(self.sparseness > 0.0):
            self.cross_entropy += (1.0 + self.sparseness) * self.sparse_fact * tf.reduce_sum(tf.multiply(self.costmask.astype(np.float32),tf.square(self.var_list[0])))
            self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate*out_dim*self.learning_rate).minimize(self.cross_entropy)
            print('Increase Sparseness')
            print(self.cross_entropy )

