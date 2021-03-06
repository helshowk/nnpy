#!/usr/bin/env python2


import time
import logging, os, sys
dir = os.path.dirname(__file__)
import cPickle as pickle

import numpy
#import cudarray as ca
#import activation, cost
import backend

logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(dir,'net.log'), level=logging.DEBUG, 
                            format='%(asctime)s %(name)s(%(levelname)s) %(module)s,%(funcName)s,%(lineno)s: %(message)s', 
                            datefmt='%m/%d/%Y %I:%M:%S %p')

class NN:
    def __init__(self, cost_fn, dcost_fn, backend_type='numpy'):
        self.correct_output = None
        self.layers = list()
        self.cost = cost_fn
        self.dcost = dcost_fn
        self.learn_rate = 0.05
        self.momentum = 0.5
        self.prev_cw = None
        self.prev_cb = None
        self.adaptive_increase = 0.025
        self.adaptive_range = [0.01, 50]
        self.current_step = 0
        self.total_cb = None
        self.total_cw = None
        self.xw_t = None
        self.xb_t = None
        self.rms_memory = 0.9
        self.l1_coefficient = 0
        self.l2_coefficient = 0
        self.ad_rho = 0.9
        self.back = backend.backends(backend_type)
        self.backlib = backend_type
        self.gradient_clip = -1
    
    def resetRMS(self):
        self.total_cb = None
        self.total_cw = None
        self.xw_t = None
        self.xb_t = None
    
    def __repr__(self):
        ret = list()
        for idx,l in enumerate(self.layers):
            ret.append("\nLayer " + str(idx) + ": ")
            ret.append(str(l))
        return '\n'.join(ret)
    
    def shape(self):
        ret = list()
        for idx,l in enumerate(self.layers):
            ret.append("Layer " + str(idx) + ": " + l.shape())
        return '\n'.join(ret)
    
    def reset(self):
        for l in self.layers:
            l.reset()
    
    def addLayer(self, l):
        self.layers.append(l)
    
    def forward(self, x, train=False):
        # forward pass input x and get output
        count = 0
        for l in self.layers:
            x = l.fwd(x, train)
            count += 1
        self.output = x
        #print "output: " + str(self.output)
        return self.output

    def clone(self):
        # return a clone of the network without copying weights, i.e. just architecture
        n = NN(self.cost, self.dcost, self.backlib)
        n.learn_rate = self.learn_rate
        n.momentum = self.momentum
        n.adaptive_increase = self.adaptive_increase
        n.adaptive_range = self.adaptive_range
        n.l1_coefficient = self.l1_coefficient
        n.l2_coefficient = self.l2_coefficient
        n.ad_rho = self.ad_rho
        for l in self.layers:
            n.addLayer(l.clone())
        return n 

    def train(self, x, t, update_count=0):
        # backprop input x with target t
        # return delta weight matrices per layer i.e. a list of delC / delW matrices and error
        y = self.forward(x, train=True)
        errors = self.cost(y, t)
        back_vec = self.dcost(y, t)
        
        #if numpy.average(errors) > 10:
            #print "errors > 10"
            #raw_input()
        
        # note these quantities are only returned, for performance purposes they can be removed
        if self.l2_coefficient <> 0:
            l2_total = 0
            for l in self.layers:
                if l.W is not None:
                    l2_total += self.back.sum(self.back.multiply(l.W,l.W))
            errors = errors + self.l2_coefficient * l2_total
            
        if self.l1_coefficient <> 0:
            l1_total = 0
            # don't apply L1 to input->first layer connections?  Regularization only on hidden connections to encourage 
            # sparsity of activations and representation versus masking inputs
            for l in self.layers[1:]:
                if l.W is not None:
                    l1_total += self.back.sum(self.back.absolute(l.W))
            errors = errors + self.l1_coefficient * l1_total
        
        delta_cw = list()
        delta_cb = list()
        for l in reversed(self.layers):
            dw, db, back_vec = l.backprop(back_vec)
            if self.l2_coefficient <> 0 and l.W is not None:
                dw += self.l2_coefficient * 2 * l.W
            
            if self.l1_coefficient <> 0 and l.W is not None:
                adj_matrix = self.l1_coefficient * (l.W >0) - self.l1_coefficient * (l.W < 0) + 0. * (l.W == 0)
                dw += adj_matrix
            
            delta_cb.insert(0,db)
            delta_cw.insert(0,dw)
        return delta_cw, delta_cb, errors, y
    
    def update(self, updateType, delta_cw, delta_cb):
        if (delta_cw is None):
            return
            
        if self.gradient_clip <> -1:
            delta_cw = [ self.back.clip(cw, -1*self.gradient_clip, self.gradient_clip) for cw in delta_cw ]
            delta_cb = [ self.back.clip(cb, -1*self.gradient_clip, self.gradient_clip) for cb in delta_cb ]
        
        if updateType == 'vanilla':
            self._update(delta_cw, delta_cb)
        elif updateType == 'adaptive':
            self._updateAdaptive(delta_cw, delta_cb)
        elif updateType == 'adagrad':
            self._updateADAGRAD(delta_cw, delta_cb)
        elif updateType == 'adadelta':
            self._updateADADELTA(delta_cw, delta_cb)
        elif updateType == 'xo':
            self._updateXO(delta_cw, delta_cb)
            
    def _update(self, delta_cw, delta_cb):
        # simplest update rule, just use momentum and learning rate
        for idx, l in enumerate(self.layers):
            # skip layers with no backprop (e.g Pooling)
            if l.W is None: continue
            
            if delta_cb[idx].shape <> l.V_B.shape:
                delta_cb[idx] = self.back.reshape(delta_cb[idx], l.V_B.shape)
            
            l.V_W = self.momentum * l.V_W - self.learn_rate * delta_cw[idx]
            l.V_B = self.momentum * l.V_B - self.learn_rate * delta_cb[idx]
            #print "\tUPDATE " + str(l.W.shape) + ", " + str(numpy.mean(l.W)) + ", " + str(numpy.mean(l.V_W)) + ", " + str(numpy.mean(delta_cw[idx]))
            l.W += l.V_W
            #print "\tUPDATE " + str(l.W.shape) + ", " + str(numpy.mean(l.W))
            l.B += l.V_B
    
    def _updateAdaptive(self, delta_cw, delta_cb):
        # update using adaptive learning rates
        for idx,l in enumerate(self.layers):
            # algorithm for both weights and bias:
            #       1.  Calculate change in gradient direction
            #       2.  If it's the same direction then increase learning rate (self.adapative_increase), otherwise decrease learning rate
            #       3.  Clip gains matrix so that we don't explode or go to zero
            
            if delta_cw[idx] is None: continue

            if self.prev_cw:
                direction = self.back.multiply(self.prev_cw[idx], delta_cw[idx])
                temp = (direction<0) * (1-self.adaptive_increase) + (direction>=0)*1
                l.G_W = self.back.multiply(l.G_W,temp)
                temp = ((direction>=0)*self.adaptive_increase + (direction < 0) * 0)
                l.G_W += temp
                l.G_W = self.back.clip(l.G_W, self.adaptive_range[0], self.adaptive_range[1])
            if self.prev_cb:
                direction = self.back.multiply(self.prev_cb[idx], delta_cb[idx])
                l.G_B = self.back.multiply(l.G_B,(direction<0) * (1-self.adaptive_increase) + (direction>=0)*1)
                l.G_B += (direction>=0)*self.adaptive_increase
                l.G_B = self.back.clip(l.G_B, self.adaptive_range[0], self.adaptive_range[1])
 
            l.V_W = self.momentum * l.V_W - self.back.multiply(self.learn_rate * l.G_W, delta_cw[idx])
            l.V_B = self.momentum * l.V_B - self.back.multiply(self.learn_rate * l.G_B, delta_cb[idx])
            
            l.W += l.V_W
            l.B += l.V_B
        
        self.prev_cw = delta_cw
        self.prev_cb = delta_cb
            
    def _updateADAGRAD(self, delta_cw, delta_cb):        
        # initialize l2 norms first
        if self.total_cw is None:
            self.current_step = 0
            self.total_cw = list(delta_cw)
            for i in range(0,len(delta_cw)):
                self.total_cw[i] = self.back.multiply(delta_cw[i], delta_cw[i])
        
        if self.total_cb is None:
            self.total_cb = list(delta_cb)
            for i in range(0,len(delta_cb)):
                self.total_cb[i] = self.back.multiply(delta_cb[i], delta_cb[i])
        
        self.current_step += 1
        for idx,l in enumerate(self.layers):
            if delta_cw[idx] is None: continue
            # compute square root of running average
            # divide weights by the square root of the running average

            # on the first compute self.total_cw[idx] == ca.multiply(delta_cw[idx], delta_cw[idx]) so this is no problem
            self.total_cw[idx] += self.back.multiply(delta_cw[idx], delta_cw[idx])
            self.total_cb[idx] += self.back.multiply(delta_cb[idx], delta_cb[idx])
        
            r_t = self.back.sqrt(self.total_cw[idx])
            r_t[numpy.where(r_t == 0)] = 1e-10
            l.V_W =  self.back.multiply(-self.learn_rate * (r_t ** -1), delta_cw[idx])
            l.W += l.V_W
            
            r_t = self.back.sqrt(self.total_cb[idx])
            r_t[numpy.where(r_t == 0)] = 1e-10            
            l.V_B = self.back.multiply(-self.learn_rate * (r_t ** -1), delta_cb[idx])
            l.B += l.V_B
            
    def _updateADADELTA(self, delta_cw, delta_cb):        
        if self.total_cw is None:
            self.total_cw = list(delta_cw)
            self.xw_t = list(delta_cw)
            for i in range(0,len(delta_cw)):
                if delta_cw[i] is not None:
                    self.total_cw[i] = self.back.zeros(delta_cw[i].shape)
                    self.xw_t[i] = self.back.zeros(delta_cw[i].shape)
                else:
                    self.total_cw[i] = None
                    self.xw_t[i] = None
        
        if self.total_cb is None:
            self.total_cb = list(delta_cb)
            self.xb_t = list(delta_cb)
            for i in range(0,len(delta_cb)):
                if delta_cb[i] is not None:
                    self.total_cb[i] = self.back.zeros(delta_cb[i].shape)
                    self.xb_t[i] = self.back.zeros(delta_cb[i].shape)
                else:
                    self.total_cb[i] = None
                    self.xb_t[i] = None
        
        for idx,l in enumerate(self.layers):
            if delta_cw[idx] is None: continue
            
            self.total_cw[idx] = self.total_cw[idx] * self.ad_rho + self.back.multiply(delta_cw[idx], delta_cw[idx])  * (1-self.ad_rho)
            self.total_cb[idx] = self.total_cb[idx] * self.ad_rho + self.back.multiply(delta_cb[idx], delta_cb[idx]) * (1-self.ad_rho)            

            r_t = self.back.sqrt(self.total_cw[idx] + 1e-6)
            rms_xw_t = self.back.sqrt(self.back.multiply(self.xw_t[idx], self.xw_t[idx]) + 1e-6)          
         
            l.V_W =  self.back.multiply(-rms_xw_t * (r_t ** -1), delta_cw[idx])
            
            
            self.xw_t[idx] = self.xw_t[idx] * self.ad_rho + self.back.multiply(l.V_W, l.V_W) * (1-self.ad_rho)
            if l.V_W.shape <> l.W.shape:
                l.V_W = self.back.reshape(l.V_W, l.W.shape)
            l.W += l.V_W
            
            r_t = self.back.sqrt(self.total_cb[idx] + 1e-6)
            rms_xb_t = self.back.sqrt(self.back.multiply(self.xb_t[idx], self.xb_t[idx]) + 1e-6)
            #rms_xb_t = self.back.abs(self.xb_t[idx]) + 1e-6
            
            l.V_B = self.back.multiply(-rms_xb_t * (r_t ** -1), delta_cb[idx])
            self.xb_t[idx] = self.xb_t[idx] * self.ad_rho + self.back.multiply(l.V_B, l.V_B) * (1-self.ad_rho)
            
            if l.V_B.shape <> l.B.shape:
                l.V_B = self.back.reshape(l.V_B, l.B.shape)
            l.B += l.V_B

    def gradient_check(self, x, y, epsilon=1e-4):
        print x.shape
        print y.shape
        # run a gradient check on this network using a random weight
        layer = None
        while (layer is None):
            layer_idx = numpy.random.random_integers(0,len(self.layers)-1)
            layer = self.layers[layer_idx]
            if layer.W is None:
                layer = None

        weight_idx = [ numpy.random.randint(dim) for dim in layer.W.shape ]
        bias_idx = [ numpy.random.randint(dim) for dim in layer.B.shape ]

        output = self.forward(x)
        dw, db, _, _ = self.train(x, y)        

        print "layer: " + str(layer_idx)
        print "weight: " + str(weight_idx)
        print "bias: " + str(bias_idx)

        print "dw:"
        print "\t" + str(dw[layer_idx][weight_idx])

        old_W = layer.W[weight_idx]
        old_B = layer.B[bias_idx]

        layer.W[weight_idx] += epsilon
        temp = self.forward(x)
        gc_plus = self.cost(temp, y)
        print '\tgc_plus: ' + str(gc_plus)

        layer.W[weight_idx] -= 2*epsilon
        temp = self.forward(x)
        gc_minus = self.cost(temp, y)
        print '\tgc_minus: ' + str(gc_minus)

        estimate = (gc_plus - gc_minus) * ((2*epsilon)**-1)
        print "\t" + str(estimate)

        print "\ndb:"
        db[layer_idx] = numpy.reshape(db[layer_idx], (layer.B.shape))
        print bias_idx
        print db[layer_idx].shape
        if len(bias_idx) > 2:
            print "\t" + str(db[layer_idx][bias_idx[0], bias_idx[1], bias_idx[2], bias_idx[3]])
        else:
            print "\t" + str(db[layer_idx][0, bias_idx[1]])

        layer.B[bias_idx] += epsilon
        temp = self.forward(x)
        gc_plus = self.cost(temp, y)
        print '\tgc_plus: ' + str(gc_plus)

        layer.B[bias_idx] -= 2*epsilon
        temp = self.forward(x)
        gc_minus = self.cost(temp, y)
        print '\tgc_minus: ' + str(gc_minus)

        estimate = (gc_plus - gc_minus) * ((2*epsilon)**-1)
        print "\t" + str(estimate)
        print "\n"

    def diagnostic(self):
        def stat_display(stat):
            return "%1.3e\t%1.3e\t%1.3e\t%1.3e" % (numpy.average(stat), numpy.std(stat), numpy.min(stat), numpy.max(stat))
        
        print "\n"
        print "-"*80
        print "parameter stats are displayed as (avg, std, min, max)"
        print "-"*80
        for idx, l in enumerate(self.layers):
            if l.is_degenerate():
                print "%s\t\tDEGENERATE"
            else:
                print "Layer %s" % (idx)
                print "W:\t%s" % (stat_display(l.W))
                print "B:\t%s" % (stat_display(l.B))
                print "V_W:\t%s" % (stat_display(l.V_W))
                print "V_B:\t%s" % (stat_display(l.V_B))
                print "G_W:\t%s" % (stat_display(l.G_W))
                print "G_B:\t%s" % (stat_display(l.G_B))
                print "act:\t%s" % (stat_display(l.activation))
                print "d_act:\t%s" % (stat_display(l.active_prime))
                print "-"*80
                
        print "\n"

    def pickle(self, location):
        # only pickling layers so if you want to re-train you might have to refactor this code, create a params dict or something like that
        out_file = open(location, 'wb')
        layer_params = [ (l.W, l.B, l.V_W, l.V_B, l.G_W, l.G_B) for l in self.layers ]
        params = dict()
        params['layers'] = layer_params
        params['total_cw'] = self.total_cw
        params['total_cb'] = self.total_cb
        params['prev_cw'] = self.prev_cw
        params['prev_cb'] = self.prev_cb
        params['xw_t'] = self.xw_t
        params['xb_t'] = self.xb_t
        
        pickle.dump(params, out_file)
        out_file.close()
    
    def load(self, location):
        in_file = open(location, 'rb')
        params = pickle.load(in_file)
        in_file.close()
        
        layer_params = params['layers']
        self.total_cw = params['total_cw']
        self.total_cb = params['total_cb']
        self.prev_cw = params['prev_cw']
        self.prev_cb = params['prev_cb']
        self.xw_t = params['xw_t']
        self.xb_t = params['xb_t']
        
        for idx, l in enumerate(self.layers):
            l.W = layer_params[idx][0]
            l.B = layer_params[idx][1]
            l.V_W = layer_params[idx][2]
            l.V_B = layer_params[idx][3]
            l.G_W = layer_params[idx][4]
            l.G_B = layer_params[idx][5]




 
