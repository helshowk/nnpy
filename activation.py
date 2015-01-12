#!/usr/bin/env python2

import os
import math
import numpy
import cudarray as ca

#nn_backend = os.getenv('NN_BACKEND', '')
#if nn_backend == 'numpy' or nn_backend == '':
#    cback = numpy
#elif nn_backend == 'cudarray':
#cback = ca

def flatten(x):
    # flatten out to avoid overflow conditions
    x[numpy.where(x > 750)] = 750
    x[numpy.where(x < -750)] = 750
    return x
    
def safe_logistic(x):
    x = flatten(x)
    #return (1. / (1. + numpy.exp(-x)))
    return (1. / (1. + ca.exp(-x, axis=1, keepdims=True)))

def d_safe_logistic(x):
    l = safe_logistic(x)
    #return numpy.multiply(l,(1-l))
    return ca.multiply(l, (1-l))

def logistic(x):
    #return (1. / (1. + ca.exp(-x)))
    return ca.nnet.sigmoid(x)

def d_logistic(x):
    l = logistic(x)
    #return numpy.multiply(l,(1-l))
    return ca.multiply(l, (1-l))

def tanh(x):
	#return numpy.tanh(x)
    return ca.tanh(x)

def d_tanh(x):
	#return 1 - numpy.power(numpy.tanh(x), 2)
    return 1 - ca.power(ca.tanh(x),2)

def softmax(x):
	# assume x is a vector of all the values
	# sum is the sum needed in denominator (i.e. sum(exp(z_j)) )
    #xmax = numpy.max(x, axis=1)
    xmax = ca.amax(x, axis=1, keepdims=True)
    #s = numpy.sum(numpy.exp(x-xmax), axis=1)
    s = ca.sum(ca.exp(x-xmax), axis=1, keepdims=True)
    #return (numpy.exp(x-xmax) / s)
    return (ca.exp(x-xmax) * (s ** -1))
    #return ca.nnet.softmax(x)
    #e = ca.exp(x)
    #return e/ca.sum(e, axis=1, keepdims=True)
    #return ca.nnet.softmax(x)

def d_softmax(x):
    y = softmax(x)
    #return numpy.multiply(y,(1-y))
    return ca.multiply(y, (1-y))

def identity(x):
    return x

def one(x):
    #return numpy.ones(x.shape)
    return ca.ones(x.shape)
    
def rectified(x):
    #temp = numpy.copy(x)
    #temp[numpy.where(temp < 0)] = 0
    #return 0*(x<0)+x*(x>=0)
    return ca.nnet.relu(x)

def d_rectified(x):
    #temp = numpy.copy(x)
    #temp[numpy.where(temp < 0)] = 0
    #temp[numpy.where(temp > 0)] = 1
    #return 0*(x<0)+1*(x>=0)
    return ca.nnet.relu_d(x)
    
def leaky(x):
    #return numpy.multiply((1*(x>0) + 1e-6 * (x<=0)),x)
    return ca.multiply((1*(x>0) + 1e-6 * (x<=0)), x)

def d_leaky(x):
    return 1*(x>0) + 1e-6*(x<=0)

def softplus(x):
    #return numpy.log(1+numpy.exp(x))
    return ca.log(1+ca.exp(x))

def d_softplus(x):
    return logistic(x)
