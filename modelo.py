import tensorflow as tf
import numpy as np

"""
The point for using truncated normal is to overcome saturation of tome 
functions like sigmoid (where if the value is too big/small, the neuron 
stops learning).
"""
def weight_variable(shape,seed):
    initial = tf.truncated_normal(shape, stddev=0.1, seed=seed)
    return tf.Variable(initial)
"""
os dados da Variable estarão diretamente ligados ao aprendizado( algoritmos de 
Machine learning). Então, sempre que criamos modelos que precisam derivar os 
dados, estes dados serão armazenados em Variable. 
"""
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class Model:
    def __init__(self, x, y_, sparseness = 0.0, learning_rate = 0.5, 
                 sparse_fact = 1.0, seed = 0):

        in_dim = int(x.get_shape()[1]) 
        out_dim = int(y_.get_shape()[1]) 

        self.x = x 
        self.y_ = y_
        self.learning_rate = learning_rate
        self.sparse_fact = sparse_fact 
        self.sparseness = sparseness
        
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.W1 = weight_variable([in_dim,out_dim],seed)
        self.b1 = bias_variable([out_dim])
    
        
        self.mask = np.ones(self.W1.get_shape()).astype(np.float32)
        if(sparseness>0.0):
            self.costmask = (np.random.random(self.W1.get_shape())<sparseness).astype(np.float32)
            self.costmask /= np.sum(self.costmask)
            self.mask[self.costmask>0] = 0.0
        
       
        self.y =  tf.matmul(x,tf.multiply(self.W1,self.mask)) + self.b1 
        self.var_list = [self.W1, self.b1]
        self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(
                        labels=y_, logits=self.y))
        
        """
        O tf.reduce_sum() efetua o somatório de matrizes, ignorando o limite 
        das dimensões. 
        x = np.asarray([[1,2,1],[1,2,3]])
        tf.reduce_sum(x)     # 10
        tf.reduce_sum(x, 0)  # [2, 4, 4]
        tf.reduce_sum(x, 1)  # [4, 6]
        """
        if(sparseness>0.0):
            self.cross_entropy += sparse_fact * tf.reduce_sum(
                    tf.multiply(self.costmask.astype(np.float32),tf.square(
                            self.var_list[0])))
        
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
            self.costmask = (np.random.random( self.W1.get_shape())<self.sparseness).astype(np.float32)
            self.costmask /= np.sum(self.costmask)
            self.mask[self.costmask<0] = 0.0
            self.W1 = tf.multiply(self.W1,self.mask)

            self.cross_entropy += self.sparse_fact * tf.reduce_sum(tf.multiply(self.costmask.astype(np.float32),tf.square(self.W1)))
           
        """
        if(self.sparseness>0.0):
            self.costmask = (np.random.random( self.W1.get_shape())<self.sparseness).astype(np.float32)
            self.costmask /= np.sum(self.costmask)
            self.mask[self.costmask>0] = 0.0


            self.cross_entropy += 1.0 * tf.reduce_sum(
                    tf.multiply(self.costmask.astype(np.float32),tf.square(
                            self.W1)))
            self.train_step = tf.train.GradientDescentOptimizer(
                self.learning_rate*out_dim*self.learning_rate).minimize(
                        self.cross_entropy)
        
            self.y =  tf.matmul(self.x,tf.multiply(self.W1,self.mask)) + self.b1 
"""
    def increase_sparsness(self):
        out_dim = int(self.y.get_shape()[1])
        if(self.sparseness > 0.0):
            self.cross_entropy += (1.0 + self.sparseness) * self.sparse_fact * tf.reduce_sum(
                    tf.multiply(self.costmask.astype(np.float32),tf.square(
                            self.var_list[0])))
            self.traino_step = tf.train.GradientDescentOptimizer(
                    self.learning_rate*out_dim*self.learning_rate).minimize(
                            self.cross_entropy)
   

