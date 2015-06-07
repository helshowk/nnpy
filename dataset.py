#!/usr/bin/env python2

import numpy as np
from backend import backends

class dataset:
    def __init__(self, x, y=None, transformer=None, backend_type='numpy'):
        self.x = x
        self.y = y
        self._x = x
        self._y = y
        self.idx = range(0, x.shape[0])
        self.position = 0
        self.transformer = transformer
        self.back = backends(backend_type)
        self.backlib = backend_type
        self.np_x = np.array(x)
        self.np_y = np.array(y)
        self.transformed = False

    def shuffle(self):
        # add resampling?
        np.random.shuffle(self.idx)
        x_copy = list(self.x)
        y_copy = list(self.y)
        for idx,i in enumerate(self.idx):
            self.x[idx] = x_copy[i]
            self.y[idx] = y_copy[i]

    def getsample(self, idx, reshape=True):
        s = self.x[idx]
        s = self.back.array(s)
        if reshape:
            s = self.back.reshape(s, ((1,) + s.shape))
        return s

    def getbatch(self,size):
        if self.position >= self.length:
            return None, None
        start_idx = self.position
        end_idx = min(self.position+size, self.length)
        self.position += size
        return self.back.array(self.x[start_idx:end_idx]), self.back.array(self.y[start_idx:end_idx])

    def transform(self):
        if not self.transformed and self.transformer is not None:
            self.x = self.transformer(self.x)
            self.transformed = True
    
    def reset(self):
        self.position = 0

    @property
    def length(self):
        return len(self.x)
