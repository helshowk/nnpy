#!/usr/bin/env python2

import numpy as np

class BaseTransformer:
    def transform(self):
        pass

class BaseNormalize:
    def __init__(self,data=None):
        self.mean = 0
        self.std = 0
        if data is not None:
            self.fit(data)

    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        

    def transform(self, data):
        self.std[self.std == 0] = 1e-10 
        return (data - self.mean) / self.std


class ChannelNormalize(BaseNormalize):
    def __init__(self,data=None):
        BaseNormalize.__init__(self,data)

    def fit(self, data):
        # data must be in a format (samples, channels, width, height)
        # this function will compute means and std and store them
        # self.mean and self.std will have dimensions (1, channels, width, height)
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        self.mean = np.reshape(self.mean, (1, data.shape[1], data.shape[2], data.shape[3]))
        self.std = np.reshape(self.std, (1, data.shape[1], data.shape[2], data.shape[3])) 
