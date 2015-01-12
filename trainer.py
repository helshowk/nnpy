#!/usr/bin/env python2

import sys, time
import numpy as np
import cudarray as ca

import neural
import activation
import utils
import os

from collections import defaultdict

class dataset:
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        self._x = x
        self._y = y
        self.idx = range(0, x.shape[0])
        self.position = 0

    def shuffle(self):
        # add resampling?
        np.random.shuffle(self.idx)

    def getbatch(self, size):
        # return batch of x, y
        start_idx = self.position
        end_idx = self.position + size
        x = None
        y = None
        indices = self.idx[start_idx:end_idx]
        x = np.vstack(self.x[indices])
        y = np.vstack(self.y[indices])

        return ca.array(x), ca.array(y)
    

class Trainer:
    # use parameters to define specific parameters for a trainer
    def __init__(self, parameters):
        self.parameters = parameters

    def snapshot(self):
        pass

class NNTrainer:
    def __init__(self, parameters):
        if not parameters.has_key['epochs']:
            parameters['epochs'] = 1
        if not parameters.has_key['updateType']:
            parameters['updateType'] = 'sgd'
        if not parameters.has_key['batchSize']:
            parameters['batchSize'] = -1
        
        self.parameters = parameters
        self.statistics = defaultdict(list)

    def train(self, network, data):
        while e < self.parameters['epochs']:
            data.shuffle()
            values, targets = data.getbatch(self.parameters['batchSize'])
            delta_w, delta_b, err, output = self.network.train(values, targets)
            network.update(self.parameters['updateType'], delta_w, delta_b) 
            self.postTrain(delta_w, delta_b, err, output)

    def postTrain(self, delta_w, delta_b, error, output):
        pass


if __name__ == "__main__":
    values = np.matrix(range(0,100)).T
    targets = np.matrix(range(0,100)).T
    test = dataset(values, targets)
    test.shuffle()
    print test.getbatch(10)
        
        
        
