#!/usr/bin/env python2

import numpy
import model, neural, cost, layer, activation

import matplotlib.pyplot as plt

if __name__ == "__main__":
    trainData = dict()
    trainData['x'] = numpy.random.rand(50000,1)*3*numpy.pi
    trainData['t'] = numpy.sin(trainData['x'])
    
    #trainData['t'] = numpy.zeros((trainData['x'].shape[0],2))
    #for idx,x in enumerate(trainData['x']):
        #trainData['t'][idx][0] = numpy.sin(x)
        #trainData['t'][idx][1] = 0.5*numpy.sin(x) + 2
    
    avg = numpy.average(trainData['x'])
    std = numpy.std(trainData['x'])
    
    network = neural.NN(cost.quadratic, cost.dquadratic)
    #network.l2_coefficient = 0.0001
    network.learn_rate = 0.005
    network.momentum = 0.5
    network.addLayer(layer.Layer(1, 100, activation.rectified, activation.d_rectified))
    #network.addLayer(layer.Layer(100, 100, activation.rectified, activation.d_rectified))
    network.addLayer(layer.Layer(100, 1, activation.identity, activation.one))
    
    #network.dropout_p = 0.5
    
    testData = dict()
    testData['x'] = numpy.random.rand(1000,1)*3*numpy.pi
    testData['t'] = numpy.sin(testData['x'])
    
    #testData['t'] = numpy.zeros((testData['x'].shape[0],2))
    #for idx,x in enumerate(testData['x']):
        #testData['t'][idx][0] = numpy.sin(x)
        #testData['t'][idx][1] = 0.5*numpy.sin(x) + 2
    
    m = model.Model(network, notes='Sin()')
    m.runModel(10, trainData, testData, updateType='', early_stopping=-1, validation=0, batchSize=100, screen_debug=True, normalize=False)

    xr = numpy.random.rand(200,1)*3*numpy.pi
    y1 = list()
    #y2 = list()
    ##for i in trainData['x']:
        ##y1.append(network.forward(i)[0])
        ##y2.append(network.forward(i)[1])

    for i in xr:
        result = network.forward(i)[0]
        y1.append(result[0])
        #y2.append(result[1])
    
    plt.plot(trainData['x'],trainData['t'], 'ro')
    ##plt.plot(xr,ans, 'ro')
    plt.plot(xr,y1, 'bo')
    #plt.plot(xr,y2, 'go')
    plt.show()

