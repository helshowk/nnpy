#!/usr/bin/env python2

import math
import numpy
import backend

class cost:
    def __init__(self,backend_type='numpy'):
        self.backlib = backend_type
        self.back = backend.backends(backend_type)

    def quadratic(self,y,t):
        return 0.5 * self.back.sum(self.back.power(t-y,2), axis=1)

    def dquadratic(self,y,t):
        return (y-t)
    
    def cross_entropy(self,y,t):
        y = 1e-6*(y == 0) + y
        return -1 * (self.back.sum(self.back.multiply(self.back.log(y), t), axis=1))
	
    def dcross_entropy(self,y,t):
    	return y - t

    #def cross_entropy_correct(self,y,t):
    #    output = self.back.zeros(y.shape)
    #    max_y = self.back.amax(y, axis=1, keepdims=True)
    #    output[numpy.where(y == max_y)] = 1
    #
    #    return self.back.array(numpy.all(t == output, axis=1))
