#!/usr/bin/env python2

import layer
import activation
import numpy

test = layer.CNLayer(1,2,10,10,5,activation.tanh,activation.d_tanh)
temp_in = numpy.random.rand(10,10)
print test.fwd([temp_in])

