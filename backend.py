#!/usr/bin/env python2

import os
os.environ['CUDARRAY_BACKEND'] = 'cuda'

import numpy as np
import cudarray as ca

def backends(backend_type):
    if backend_type == 'numpy':
        return np
    if backend_type == 'cudarray':
        return ca
