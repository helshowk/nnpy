#!/usr/bin/env python2

import os
import math
import numpy
#import cudarray as ca
import backend


class activation:
    def __init__(self, backend_type='numpy'):
        self.backlib = backend_type
        self.back = backend.backends(backend_type)

    def clip(self,x):
        # flatten out to avoid overflow conditions
        x[numpy.where(x > 750)] = 750
        x[numpy.where(x < -750)] = 750
        return x
    
    def safe_logistic(self,x):
        x = self.clip(x)
        return (1. / (1. + self.back.exp(-x, axis=1, keepdims=True)))

    def d_safe_logistic(self, x):
        l = self.safe_logistic(x)
        return self.back.multiply(l, (1-l))

    def logistic(self, x):
        if self.backlib == 'numpy':
            return (1. / (1. + self.back.exp(-x)))
        elif self.backlib == 'cudarray':
            return self.back.nnet.sigmoid(x)

    def d_logistic(self, x):
        l = self.logistic(x)
        return self.back.multiply(l, (1-l))

    def tanh(self, x):
        return self.back.tanh(x)

    def d_tanh(self,x):
        return 1 - self.back.power(self.back.tanh(x),2)

    def softmax(self, x):
	    # assume x is a vector of all the values
    	# sum is the sum needed in denominator (i.e. sum(exp(z_j)) )
        #xmax = numpy.max(x, axis=1)
        if self.backlib == 'numpy':
            xmax = self.back.max(x, axis=1, keepdims=True)
            s = self.back.sum(self.back.exp(x-xmax), axis=1, keepdims=True)
            return (self.back.exp(x-xmax) / s)
        elif self.backlib == 'cudarray':
            return self.back.nnet.softmax(x)
        #e = ca.exp(x)
        #return e/ca.sum(e, axis=1, keepdims=True)
        #return ca.nnet.softmax(x)

    def d_softmax(self, x):
        y = self.softmax(x)
        return self.back.multiply(y, (1-y))

    def identity(self,x):
        return x

    def one(self,x):
        #return numpy.ones(x.shape)
        return self.back.ones(x.shape)
    
    def rectified(self, x):
        #temp = numpy.copy(x)
        #temp[numpy.where(temp < 0)] = 0
        #return 0*(x<0)+x*(x>=0)
        if self.backlib == 'cudarray':
            return self.back.nnet.relu(x)
        elif self.backlib == 'numpy':
            temp = numpy.copy(x)
            temp[numpy.where(temp < 0)] = 0
            return 0*(x<0)+x*(x>=0)
        #self.back.copyto(x, temp)
        #temp[numpy.where(temp < 0)] = 0
        #return 0*(x<0) + x*(x>=0) 

    def d_rectified(self, x):
        if self.backlib == 'cudarray':
            return self.back.nnet.relu_d(x)
        else:
            temp = numpy.copy(x)
            temp[numpy.where(temp < 0)] = 0
            temp[numpy.where(temp > 0)] = 1
            return 0*(x<0)+1*(x>=0)
        #temp = self.back.copy(x)
        #temp[numpy.where(temp < 0)] = 0
        #temp[numpy.where(temp > 0)] = 1
        #return 0*(x<0)+1*(x>=0)
        
            
    
    def leaky(self, x):
        return self.back.multiply((1*(x>0) + 1e-6 * (x<=0)), x)

    def d_leaky(self, x):
        return 1*(x>0) + 1e-6*(x<=0)

    def softplus(self,x):
        #return numpy.log(1+numpy.exp(x))
        return self.back.log(1+self.back.exp(x))

    def d_softplus(self,x):
        return self.logistic(x)
