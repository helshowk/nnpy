#!/usr/bin/env python2

import numpy
import cudarray as ca
import activation

class CNLayer:
    # convolutional layer
    
    def __init__(self, input_maps, output_maps, input_rows, input_cols, field_dim, f, df):
        self.W = list()         # first index is from input maps, second is to output maps, value is a weight matrix
        
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.field_dim = field_dim
        self.f = f
        self.df = f

        self.output_map_rows = self.input_rows - self.field_dim
        self.output_map_cols = self.input_cols - self.field_dim
        self.reset()
        
    def reset(self):
        self.B = ca.zeros((1,self.output_maps))
        
        for i in range(0, self.input_maps):
            self.W.append([])
            for j in range(0, self.output_maps):
                self.W[i].append(ca.random.normal(0,0.1,(self.field_dim, self.field_dim)))

    def convolve(self, input_map, field):
        # apply field over the input map and return a matrix
        
        result = ca.zeros((self.output_map_rows, self.output_map_cols))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                end_i = i + field.shape[0]
                end_j = j + field.shape[1]
                idx_array = ca.array(range(i, end_i))
                idx_array2 = ca.array(range(j, end_j))
                #print input_map[idx_array, idx_array2]
                #print input_map[numpy.array(range(i,end_i)), numpy.array(range(j,end_j))]
                result[i][j] = ca.sum(ca.multiply(input_map[idx_array[:, None], idx_array2], field))
        
        return result

    def fwd(self, x, train=False):
        # x is a list of input maps
        # result is a list of output maps
        self.output = list()
        self.d_output = list()
        
        for j in range(0, self.output_maps):
            total = ca.zeros((self.output_map_rows, self.output_map_cols))
            for idx, input_map in enumerate(x):
                weight_field = self.W[idx][j]
                total += self.convolve(input_map, weight_field)
            
            print total
            self.output.append(self.f(total + self.B[0][j]))
            self.d_output.append(self.df(total + self.B[0][j]))
        
        self.active_prime = [ ca.array(do) for do in self.d_output ]
        self.activation = [ ca.array(o) for o in self.output ]
        
        return self.activation
        
class Layer:
    # weight matrix is nxm and represents incoming connections from previous n nodes
    def __init__(self, n, m, f, df, ones=False, dropout_p = 1, input_dropout_p = 1):
        # activation function and derivative
        self.f = f
        self.df = df
        # store dimensions of layer
        self.n = n
        self.m = m
        # dropout
        self.dropout_p = dropout_p
        self.input_dropout_p = input_dropout_p

        self.cudarray_cnt = 0

        self.reset()

        
    def __repr__(self):
        return "Activation: " + str(self.f) + "\nDropout: " + str(self.dropout_p) + "\nShape: (" + str(self.W.shape) + ")\nBias: " + str(self.B) + "\nWeights: \n" + str(self.W)
    
    
    def reset(self):
        # reset matrices
        # weight matrix
        #self.W = (2.4/numpy.sqrt(self.n))*numpy.random.normal(0,1,(self.n, self.m))
        self.W = (1/numpy.sqrt(self.n))*ca.random.normal(0,1,(self.n, self.m))
        self.W = ca.array(self.W)
        # velocity matrix
        self.V_W = ca.zeros((self.n, self.m))
        
        # self.B could become a fixed part of self.W
        # bias vector values
        if self.f == activation.rectified:
            self.B = 0.1 * ca.ones((1, self.m))
        else:
            #self.B = (2.4/numpy.sqrt(self.n))*numpy.random.normal(0,1,(1, self.m))
            self.B = (1/numpy.sqrt(self.n))*ca.random.normal(0,1,(1, self.m))
            self.B = ca.array(self.B)
        #  bias vector velocity matrix
        self.V_B = ca.zeros((1, self.m))
        
        # Adaptive learning rates
        # gains on matrix of learning rates for weights and bias
        self.G_W = ca.ones((self.n, self.m))
        self.G_B = ca.ones((1, self.m))
        
        self.logit = None
        self.activation = None
        self.active_prime = None    


    def shape(self):
        return str(self.W.shape)
    
    def dropout(self, shape):
        if self.dropout_p == 1:
            return ca.ones(shape)
        else:
            # bernoulli
            # dropout neurons
            sample = ca.random.rand(shape[0], shape[1])
            return sample > (1-self.dropout_p)
    
    def fwd(self, x, train=False):
        # forward pass input x through this layer and return activations stored in self.activations
        # activation derivatives are stored in self.activation_prime
        # logit values are also stored in self.logit
        
        #print "layer"
        #print x
        self.input_values = x
        if train and self.input_dropout_p <> 1:
            dropout_matrix = self.dropout(self.input_values.shape)
            ca.multiply(self.input_values, dropout_matrix, out=self.input_values)
        else:
            ca.multiply(self.input_values,self.input_dropout_p, out=self.input_values)
       
        if self.logit is None: 
            self.cudarray_cnt += 1
            self.logit = ca.zeros((x.shape[0], self.W.shape[1]))
        
        #print "Parameters"
        #print self.W 
        #print self.B
        ca.dot(self.input_values,self.W, out=self.logit)
        self.logit += self.B
        #active_f = self.f(self.logit)
        #active_df = self.df(self.logit)

        if self.activation is None:
            self.cudarray_cnt += 1
            self.activation = ca.zeros((x.shape[0], self.W.shape[1]))

        if self.active_prime is None:
            self.cudarray_cnt += 1
            self.active_prime = ca.zeros((x.shape[0], self.W.shape[1]))

        #print "Logit"
        #print self.logit 
        #print "pre-activation"
        #print self.activation
        #print self.f(self.logit)
        self.activation = self.f(self.logit)
        self.active_prime = self.df(self.logit)

        if train and self.dropout_p <> 1:
            dropout_matrix = self.dropout(active_f.shape)
            ca.multiply(self.activation, dropout_matrix, out=self.activation)
            ca.multiply(self.active_prime, dropout_matrix, out=self.active_prime)
        else:
            self.activation *= self.dropout_p
            self.active_prime *= self.dropout_p

        #print "activation"        
        #print self.activation
        return self.activation

