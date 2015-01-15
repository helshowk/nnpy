#!/usr/bin/env python2

import numpy
#import cudarray as ca
#import activation
import backend


class CNLayer:
    # convolutional layer
    
    def __init__(self, input_maps, output_maps, input_rows, input_cols, field_dim, f, df, backend_type='numpy'):
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
        self.back = backend.backends(backend_type)
        self.backlib = backend_type
        
    def reset(self):
        self.B = self.back.zeros((1,self.output_maps))
        
        for i in range(0, self.input_maps):
            self.W.append([])
            for j in range(0, self.output_maps):
                self.W[i].append(self.back.random.normal(0,0.1,(self.field_dim, self.field_dim)))

    def convolve(self, input_map, field):
        # apply field over the input map and return a matrix
        
        result = self.back.zeros((self.output_map_rows, self.output_map_cols))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                end_i = i + field.shape[0]
                end_j = j + field.shape[1]
                idx_array = self.back.array(range(i, end_i))
                idx_array2 = self.back.array(range(j, end_j))
                result[i][j] = self.back.sum(self.back.multiply(input_map[idx_array[:, None], idx_array2], field))
        
        return result

    def fwd(self, x, train=False):
        # x is a list of input maps
        # result is a list of output maps
        self.output = list()
        self.d_output = list()
        
        for j in range(0, self.output_maps):
            total = self.back.zeros((self.output_map_rows, self.output_map_cols))
            for idx, input_map in enumerate(x):
                weight_field = self.W[idx][j]
                total += self.convolve(input_map, weight_field)
            
            self.output.append(self.f(total + self.B[0][j]))
            self.d_output.append(self.df(total + self.B[0][j]))
        
        self.active_prime = [ self.back.array(do) for do in self.d_output ]
        self.activation = [ self.back.array(o) for o in self.output ]
        
        return self.activation
        
class Layer:
    # weight matrix is nxm and represents incoming connections from previous n nodes
    def __init__(self, n, m, f, df, ones=False, dropout_p = 1, input_dropout_p = 1, backend_type='numpy'):
        # activation function and derivative
        self.f = f
        self.df = df
        # store dimensions of layer
        self.n = n
        self.m = m
        # dropout
        self.dropout_p = dropout_p
        self.input_dropout_p = input_dropout_p

        self.back = backend.backends(backend_type)
        self.backlib = backend_type
        self.reset()

        
    def __repr__(self):
        return "Activation: " + str(self.f) + "\nDropout: " + str(self.dropout_p) + "\nShape: (" + str(self.W.shape) + ")\nBias: " + str(self.B) + "\nWeights: \n" + str(self.W)
    
    
    def reset(self):
        # reset matrices
        self.W = self.back.array(numpy.random.normal(0,1,(self.n, self.m)))
        self.W *= (self.n ** -0.5)
        self.V_W = self.back.zeros((self.n, self.m))
        
        self.B = 0.1 * self.back.ones((1, self.m))
        self.V_B = self.back.zeros((1, self.m))
        
        # Adaptive learning rates
        # gains on matrix of learning rates for weights and bias
        self.G_W = self.back.ones((self.n, self.m))
        self.G_B = self.back.ones((1, self.m))
        
        self.logit = None
        self.activation = None
        self.active_prime = None    


    def shape(self):
        return str(self.W.shape)
    
    def dropout(self, shape):
        if self.dropout_p == 1:
            return self.back.ones(shape)
        else:
            # bernoulli
            # dropout neurons
            sample = self.back.random.uniform(shape[0], shape[1])
            return sample > (1-self.dropout_p)
    
    def fwd(self, x, train=False):
        # forward pass input x through this layer and return activations stored in self.activations
        # activation derivatives are stored in self.activation_prime
        # logit values are also stored in self.logit
        
        self.input_values = x
        if train and self.input_dropout_p <> 1:
            dropout_matrix = self.dropout(self.input_values.shape)
            self.back.multiply(self.input_values, dropout_matrix, out=self.input_values)
        else:
            self.input_values *= self.input_dropout_p
       
        # create memory space for logit on first pass
        if self.logit is None: 
            self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))
        elif self.logit.shape <> (x.shape[0], self.W.shape[1]):
            self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))
 
        self.back.dot(self.input_values,self.W, out=self.logit)
        self.logit += self.B

        if self.activation is None:
            self.activation = self.back.zeros((x.shape[0], self.W.shape[1]))
        elif self.activation.shape <> (x.shape[0], self.W.shape[1]):
            self.activation = self.back.zeros((x.shape[0], self.W.shape[1]))

        if self.active_prime is None:
            self.active_prime = self.back.zeros((x.shape[0], self.W.shape[1]))
        elif self.active_prime.shape <> (x.shape[0], self.W.shape[1]):
            self.active_prime = self.back.zeros((x.shape[0], self.W.shape[1]))

        self.activation = self.f(self.logit)
        self.active_prime = self.df(self.logit)
        
        if train and self.dropout_p <> 1:
            dropout_matrix = self.dropout(self.activation.shape)
            self.back.multiply(self.activation, dropout_matrix, out=self.activation)
            self.back.multiply(self.active_prime, dropout_matrix, out=self.active_prime)
        else:
            self.activation *= self.dropout_p
            self.active_prime *= self.dropout_p

        return self.activation

