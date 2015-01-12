#!/usr/bin/env python2

import math
import numpy
import cudarray as ca

def quadratic(y,t):
    #return 0.5*numpy.sum(numpy.power((t-y),2))
    # I haven't tested that yet
    return 0.5*ca.sum((t-y)**2)

def dquadratic(y,t):
    return (y-t)
    
def cross_entropy(y,t):
    #y[numpy.where(y == 0)] = 1e-10
    #return -1 * (numpy.sum(numpy.multiply(numpy.log(y), t)))
    # haven't tested this either
    return -1 * (ca.sum(ca.multiply(ca.log(y), t)))
	
def dcross_entropy(y,t):
	return y - t

def cross_entropy_correct(y,t):
    output = ca.zeros(y.shape)
    max_y = ca.amax(y, axis=1, keepdims=True)
    output[numpy.where(y == max_y)] = 1

    return ca.array(numpy.all(t == output, axis=1))
