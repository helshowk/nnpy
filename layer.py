#!/usr/bin/env python2

import numpy
#import cudarray as ca
#import activation
import backend


class CNLayer:
    # convolutional layer
    
    def __init__(self, input_maps, output_maps, input_rows, input_cols, field_dim, f, df, stride=0, backend_type='numpy'):
        self.W = list()         # first index is from input maps, second is to output maps, value is a weight matrix
        
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.input_rows = input_rows
        self.input_cols = input_cols
        self.field_dim = field_dim
        self.f = f
        self.df = df
    
        self.stride = stride

        self.output_map_rows = self.input_rows - self.field_dim + 1
        self.output_map_cols = self.input_cols - self.field_dim + 1
        
        self.back = backend.backends(backend_type)
        self.backlib = backend_type

        self.reset()
        
    def reset(self):
        self.B = self.back.random.normal(0, 0.1, (1,self.output_maps))
        self.tW = 0.1 * self.back.random.uniform(size=(self.input_maps, self.output_maps, self.field_dim, self.field_dim))        

 
        # each input_map x output_map has a weight field associated with it
        for i in range(0, self.input_maps):
            self.W.append([])
            for j in range(0, self.output_maps):
                #self.W[i].append(self.back.random.normal(0,0.1,(self.field_dim, self.field_dim)))
                self.W[i].append(0.1*self.back.random.uniform(size=(self.field_dim, self.field_dim)))

    def convolve(self, input_map, field):
        # apply field over the input map and return a matrix
        rows = input_map.shape[0] - field.shape[0] + 1
        cols = input_map.shape[1] - field.shape[1] + 1
        result = self.back.zeros((rows, cols))
        for i in range(0, result.shape[0]):
            for j in range(0, result.shape[1]):
                end_i = i + field.shape[0]
                end_j = j + field.shape[1]
                idx_array = self.back.array(range(i, end_i))
                idx_array2 = self.back.array(range(j, end_j))
                result[i][j] = self.back.sum(self.back.multiply(input_map[idx_array[:, None], idx_array2], field))
        
        return result

    def backprop(self, deltas):
        # using previous step deltas compute the deltas of this layer
        self.delta_w = list()
        self.back_deltas = list()

        # if pooling then convert all deltas back to the output map size, i.e. 0 in non-selected elements
        if self.stride <> 0:
            new_deltas = list()
            if (deltas[0].shape <> (self.output_map_rows / self.stride, self.output_map_cols/self.stride)):
                temp = numpy.split(deltas[0], self.output_maps)
                deltas = [ m.reshape(self.output_map_rows/self.stride, self.output_map_cols/self.stride) for m in temp ]    
            for idx, d in enumerate(deltas):
                temp_d = numpy.zeros((self.output_map_rows, self.output_map_cols))
                for i in range(0, d.shape[0]):
                    for j in range(0, d.shape[1]):
                        masked_idx = self.masked_activation[(idx, i, j)]
                        temp_d[masked_idx] = d[i][j]
                new_deltas.append(temp_d)
            deltas = new_deltas

        zero_row = numpy.zeros((1,self.output_map_cols+4))
        zero_col = numpy.zeros((self.output_map_rows, 1))

        self.delta_b = list()

        for i in range(0, self.input_maps):
            self.delta_w.append([])
            input_map = self.input_values[i]
            back_delta = numpy.zeros(input_map.shape)

            for j in range(0, self.output_maps):
                # backprop for each kernel
                del_o_k = numpy.multiply(self.active_prime[j], deltas[j])

                # first convolution runs the f' * del_C matrix on the output map
                # over the input map to get our kernel gradients
                answer = self.convolve(input_map, del_o_k)
                self.delta_w[i].append(answer) 
                if len(self.delta_b) < self.output_maps:
                    #print '============== delta_b_cal ' + str(j) + ' =============='
                    #print self.active_prime[j]
                    #print deltas[j]
                    #print '=================================='
                    self.delta_b.append(self.back.sum(self.back.multiply(self.active_prime[j], deltas[j])))
                # second convolution runs the inverted kernel map over the padded f' * del_c
                # to produce the del_c for next step

                # start with activation prime multiplied by the output map:
                #       f' * omega^{l+1}
                # Then convolve over that using the inverted kernel
                out_prime = self.back.multiply(self.active_prime[j], deltas[j])
                k = self.W[i][j]
                inv_k = numpy.flipud(numpy.fliplr(k))
                out_prime = numpy.hstack([zero_col, zero_col, out_prime, zero_col, zero_col])
                out_prime = numpy.vstack([zero_row, zero_row, out_prime, zero_row, zero_row])
                back_delta += self.convolve(out_prime, inv_k)
            
            self.back_deltas.append(back_delta)
    
        return self.delta_w, self.delta_b, self.back_deltas

    def max_pool(self, activation):
        if self.stride == 0:
            return activation
        else:
            rows = self.output_map_rows
            cols = self.output_map_cols
            max_pooled_out = list()
            self.masked_activation = dict()
            for o in self.activation:
                max_pooled_out.append(numpy.zeros((rows/self.stride, cols/self.stride)))
                idx = len(max_pooled_out)-1
                for i in range(0,rows-1,self.stride):
                    for j in range(0,cols-1,self.stride):
                        row_idx = self.back.array(range(i,i + self.stride))
                        col_idx = self.back.array(range(j,j + self.stride))
                        stride_matrix = o[row_idx[:, None], col_idx]

                        max_value = self.back.amax(stride_matrix)
                        max_idx = numpy.where( stride_matrix == max_value)
                        max_row = max_idx[0][0] + i
                        max_col = max_idx[1][0] + j

                        self.masked_activation[(idx,i/self.stride,j/self.stride)] = (max_row, max_col)
                        max_pooled_out[idx][i/self.stride][j/self.stride] = max_value

            return max_pooled_out

    def fwd(self, x, train=False):
        # x is a list of input maps
        # result is a list of output maps
        self.output = list()
        self.d_output = list()
        self.input_values = x
 
        for j in range(0, self.output_maps):
            total = self.back.zeros((self.output_map_rows, self.output_map_cols))
            #total2 = self.back.zeros((self.output_map_rows, self.output_map_cols))
            for idx, input_map in enumerate(x):
                weight_field = self.W[idx][j]
                total += self.convolve(input_map, weight_field)
                
            # apply non-linear transform 
            #print 'basis: '
            #print self.B[j]
            #print total
            #print self.B[j] + total
            #print 'end basis'
            #print "Output: " + str(self.f(total + self.B[j]))
            self.output.append(self.f(total + self.B[0][j]))
            self.d_output.append(self.df(total + self.B[0][j]))
        
        self.active_prime = [ self.back.array(do) for do in self.d_output ]
        self.activation = [ self.back.array(o) for o in self.output ]
        
        return self.max_pool(self.activation)
        
class Layer:
    # weight matrix is nxm and represents incoming connections from previous n nodes
    def __init__(self, n, m, f, df, dropout_p = 1, input_dropout_p = 1, backend_type='numpy'):
        # activation function and derivative
        self.f = f
        self.df = df
        # store dimensions of layer
        self.n = n
        self.m = m
        # dropout
        self.dropout_p = dropout_p
        self.input_dropout_p = input_dropout_p

        self.back = backend.backends(backend_type)
        self.backlib = backend_type
        self.reset()

        
    def __repr__(self):
        return "Activation: " + str(self.f) + "\nDropout: " + str(self.dropout_p) + "\nShape: (" + str(self.W.shape) + ")\nBias: " + str(self.B) + "\nWeights: \n" + str(self.W)
    
    
    def reset(self):
        # reset matrices
        self.W = self.back.array(numpy.random.normal(0,1,(self.n, self.m)))
        self.W *= (self.n ** -0.5)
        self.V_W = self.back.zeros((self.n, self.m))
        
        self.B = 0.1 *  self.back.ones((1, self.m))
        self.V_B = self.back.zeros((1, self.m))
        
        # Adaptive learning rates
        # gains on matrix of learning rates for weights and bias
        self.G_W = self.back.ones((self.n, self.m))
        self.G_B = self.back.ones((1, self.m))
        
        self.logit = None
        self.activation = None
        self.active_prime = None    

    def clone(self):
        # return a clone without copying weight values
        l = Layer(self.n, self.m, self.f, self.df, self.dropout_p, self.input_dropout_p, self.backlib)
        return l

    def shape(self):
        return str(self.W.shape)
    
    def dropout(self, shape):
        if self.dropout_p == 1:
            return self.back.ones(shape)
        else:
            # bernoulli
            # dropout neurons
            sample = self.back.random.uniform(shape[0], shape[1])
            return sample > (1-self.dropout_p)
   
    def backprop(self, deltas):
        temp = self.back.multiply(deltas, self.active_prime)
        db = self.back.mean(temp, axis=0)
        
        dw = self.back.zeros((temp.shape[1], self.input_values.shape[1]))
        self.back.dot(temp.T, self.input_values, out=dw)
        dw = self.back.array(numpy.transpose(dw))
        self.back.multiply(dw, (self.input_values.shape[0] ** -1), out=dw)
        
        back_delta = self.back.dot(self.W, temp.T)
        back_delta = self.back.array(numpy.transpose(back_delta))
        
        return dw, db, back_delta

 
    def fwd(self, x, train=False):
        # forward pass input x through this layer and return activations stored in self.activations
        # activation derivatives are stored in self.activation_prime
        # logit values are also stored in self.logit
 
        # if input values are lists of maps then they're coming from a convolution
        # layer and we need to put them side by side as a single matrix

        self.input_values = x
        if train and self.input_dropout_p <> 1:
            dropout_matrix = self.dropout(self.input_values.shape)
            self.back.multiply(self.input_values, dropout_matrix, out=self.input_values)
        else:
            self.input_values *= self.input_dropout_p
       
        # create memory space for logit on first pass
        if self.logit is None: 
            try:
                self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))
            except AttributeError:
                # we're dealing with a convolution so modify x appropriately and reset input_values
                x = numpy.matrix(numpy.hstack([ m.ravel() for m in x]))
                self.input_values = self.back.array(x)
                self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))

        elif self.logit.shape <> (x.shape[0], self.W.shape[1]):
            try:
                self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))
            except AttributeError:
                x = numpy.matrix(numpy.hstack([ m.ravel() for m in x]))
                self.input_values = self.back.array(x)
                self.logit = self.back.zeros((x.shape[0], self.W.shape[1]))


        self.back.dot(self.input_values,self.W, out=self.logit)
        self.logit += self.B

        if self.activation is None:
            self.activation = self.back.zeros((x.shape[0], self.W.shape[1]))
        elif self.activation.shape <> (x.shape[0], self.W.shape[1]):
            self.activation = self.back.zeros((x.shape[0], self.W.shape[1]))

        if self.active_prime is None:
            self.active_prime = self.back.zeros((x.shape[0], self.W.shape[1]))
        elif self.active_prime.shape <> (x.shape[0], self.W.shape[1]):
            self.active_prime = self.back.zeros((x.shape[0], self.W.shape[1]))

        self.activation = self.f(self.logit)
        self.active_prime = self.df(self.logit)
        
        if train and self.dropout_p <> 1:
            dropout_matrix = self.dropout(self.activation.shape)
            self.back.multiply(self.activation, dropout_matrix, out=self.activation)
            self.back.multiply(self.active_prime, dropout_matrix, out=self.active_prime)
        else:
            self.activation *= self.dropout_p
            self.active_prime *= self.dropout_p

        return self.activation

