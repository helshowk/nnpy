#!/usr/bin/env python2

import os
os.environ['CUDARRAY_BACKEND'] = 'cuda'

import numpy
import cudarray as ca
import model, neural, cost, layer, activation

def l1_regularization(network):
    l1_total = 0
    for l in network.layers:
        l1_total += ca.sum(ca.absolute(l.W))
    l1_total =  l1_total * network.l1_coefficient
    return l1_total

def l2_regularization(network):
    # l2 regularization
    l2_total = 0
    for l in network.layers:    
        l2_total += ca.sum(ca.multiply(l.W,l.W))
    l2_total = l2_total * network.l2_coefficient
    return l2_total

def gradientCheck(network, epsilon, x, t):
    # returns a list of tuples with matrices of approximations per weight if i and j aren't specified
    approximations = list()
    
    # train network once
    delta_cw, delta_cb, err, _ = network.train(x,t)
    
    for idx,layer in enumerate(network.layers):
        layer_weight_approximation = ca.zeros(layer.W.shape)
        layer_bias_approximation = ca.zeros(layer.B.shape)
        
        rows = layer.W.shape[0]
        cols = layer.W.shape[1]
        # weights
        for i in range(0,rows):
            for j in range(0,cols):
                # store previous weight
                old_weight = layer.W[i][j]
                
                # compute gc+
                # forward pass with W(0,0) + epsilon and calculate cost function value
                layer.W[i][j] = old_weight + epsilon
                y = network.forward(x)
                temp = network.cost(y,t)
                gc_plus = network.cost(y, t) + l2_regularization(network) + l1_regularization(network)
                # compute gc-
                # forward pass with W(0,0) - epsilon and calculate cost function value
                layer.W[i][j] = old_weight - epsilon    
                y = network.forward(x)
                gc_minus = network.cost(y, t) + l2_regularization(network) + l1_regularization(network)
                # approximate
                gc_approx = (gc_plus - gc_minus) * ((2*epsilon)**-1)
                layer_weight_approximation[i][j] = gc_approx[0][0]
                
                # restore previous weight
                layer.W[i][j] = old_weight
        
        # BIAS
        cols = layer.B.shape[1]
        for k in range(0,cols):
            old_bias = layer.B[0][k]
            
            layer.B[0][k] = old_bias + epsilon
            y = network.forward(x)
            gc_plus = network.cost(y, t)
            
            layer.B[0][k] = old_bias - epsilon
            y = network.forward(x)
            gc_minus = network.cost(y, t)
            
            gc_approx = (gc_plus - gc_minus) / (2*epsilon)
            layer_bias_approximation[0][k] = gc_approx[0][0]
            
            layer.B[0][k] = old_bias
        
        approximations.append([layer_weight_approximation, layer_bias_approximation])
    
    return delta_cw, delta_cb, approximations

def averageGradientCheck(network, iterations, x, t, threshold=1e-8):
    # average approximations over a few iterations
    delta_cw = list()
    delta_cb = list()
    approximations = list()
    for i in range(0,iterations):
        dcw, dcb, approx = gradientCheck(network, 1e-4, x, t)
        for i in range(0,len(dcw)):
            try:
                delta_cw[i] += dcw[i]
                delta_cb[i] += dcb[i]
                approximations[i][0] += approx[i][0]
                approximations[i][1] += approx[i][1]
            except IndexError:
                delta_cw.append(dcw[i])
                delta_cb.append(dcb[i])
                approximations.append(approx[i])
    
    for i in range(0, len(delta_cw)):
        delta_cw[i] /= 50
        delta_cb[i] /= 50
        approximations[i][0] /= 50
        approximations[i][1] /= 50

    # print out results per layer:
    for idx,l in enumerate(delta_cw):
        weights = l - approximations[idx][0]
        bias = delta_cb[idx] - approximations[idx][1]
        if numpy.any(weights > threshold):
            print "FAIL on Weights"
            print weights
        if numpy.any(bias > threshold):
            print "FAIL on Bias:"
            print bias

def quadratic():
    # quadratic error network
    print "Quadratic error, 3 layer 1x5x5x1 tanh network with L2 regularization"
    network = neural.NN(cost.quadratic, cost.dquadratic)
    #network.l1_coefficient = 1
    #network.l2_coefficient = 0.1
    network.addLayer(layer.Layer(1, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 1, activation.tanh, activation.d_tanh))
    averageGradientCheck(network, 1, x, t)
    
    print "Quadratic error, 3 layer 1x5x5x1  logistic network with L2 regularization"
    #network.layers[0] = layer.Layer(1,5, activation.logistic, activation.d_logistic)
    #network.layers[1] = layer.Layer(5,5, activation.logistic, activation.d_logistic)
    #network.layers[2] = layer.Layer(5,1, activation.logistic, activation.d_logistic)
    #averageGradientCheck(network, 1, x, t)
    
    print "Quadratic error, 3 layer 1x5x5x1  rectified network with L2 regularization"
    #network.layers[0] = layer.Layer(1,5, activation.rectified, activation.d_rectified)
    #network.layers[1] = layer.Layer(5,5, activation.rectified, activation.d_rectified)
    #network.layers[2] = layer.Layer(5,1, activation.rectified, activation.d_rectified)
    #averageGradientCheck(network, 1, x, t)
    
    print "Quadratic error, 3 layer 1x5x5x1  leaky rectified network with L2 regularization"
    #network.layers[0] = layer.Layer(1,5, activation.leaky, activation.d_leaky)
    #network.layers[1] = layer.Layer(5,5, activation.leaky, activation.d_leaky)
    #network.layers[2] = layer.Layer(5,1, activation.leaky, activation.d_leaky)
    #averageGradientCheck(network, 1, x, t)
    
    print "Quadratic error, 3 layer 1x5x5x1  softplus network with L2 regularization"
    #network.layers[0] = layer.Layer(1,5, activation.softplus, activation.d_softplus)
    #network.layers[1] = layer.Layer(5,5, activation.softplus, activation.d_softplus)
    #network.layers[2] = layer.Layer(5,1, activation.softplus, activation.d_softplus)
    #averageGradientCheck(network, 1, x, t, threshold=1e-6)
    
    print "Quadratic error, 4 layer 1x5x5x5x1  logistic, tanh, rectified, identity network with L2 regularization"
    #network.layers[0] = layer.Layer(1,5, activation.logistic, activation.d_logistic)
    #network.layers[1] = layer.Layer(5,5, activation.tanh, activation.d_tanh)
    #network.layers[1] = layer.Layer(5,5, activation.rectified, activation.d_rectified)
    #network.layers[2] = layer.Layer(5,1, activation.identity, activation.one)
    #averageGradientCheck(network, 1, x, t)

def cross():
    # a postiive / negative tanh network classifier
    t = 0 if x < 0 else 1
    print "Cross-entropy, 3 layer 1x5x5x1  tanh and softmax network"
    network = neural.NN(cost.cross_entropy, cost.dcross_entropy)
    #network = neural.NN(cost.quadratic, cost.dquadratic)
    network.addLayer(layer.Layer(1, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 1, activation.softmax, activation.d_softmax))
    averageGradientCheck(network, 20, x, t)
    
    print "Cross-entropy, 3 layer 1x5x5x1  logistic and softmax network"
    network = neural.NN(cost.cross_entropy, cost.dcross_entropy)
    network.addLayer(layer.Layer(1, 5, activation.logistic, activation.d_logistic))
    network.addLayer(layer.Layer(5, 5, activation.logistic, activation.d_logistic))
    network.addLayer(layer.Layer(5, 1, activation.softmax, activation.d_softmax))
    averageGradientCheck(network, 20, x, t)
    
def multiple():
    # multiple outputs
    x = ca.array(ca.random.uniform(size=(1,1)))*2*numpy.pi
    t = ca.zeros((1,2))
    for idx,v in enumerate(x):
        t[idx][0] = ca.sin(v[0])
        t[idx][1] = ca.sin(v[0]) + 2
    
    print "Multiple outputs, Quadratic error, 3 layer 1x5x5x2 tanh network with L2 regularization"
    network = neural.NN(cost.quadratic, cost.dquadratic)
    network.l1_coefficient = 1
    #network.l2_coefficient = 0.1
    network.addLayer(layer.Layer(1, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 5, activation.tanh, activation.d_tanh))
    network.addLayer(layer.Layer(5, 2, activation.tanh, activation.d_tanh))
    averageGradientCheck(network, 20, x, t)
        
if __name__ == "__main__":
    x = ca.array(ca.random.uniform(size=(1,1)))*2*numpy.pi
    t = ca.sin(x)
    
    quadratic()
    print "\n"
    cross()
    print "\n"
    #multiple()
    
    

    
