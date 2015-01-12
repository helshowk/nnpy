#!/usr/bin/env python2

# TODO:
#   - Proper Cross-validation

import sys, time
import numpy
import cudarray as ca

import neural
import activation
import utils
import os

import json, gzip

from collections import defaultdict

numpy.set_printoptions(suppress=True)

def printConfusionMatrix(confusion_matrix):
    # output formatting for confusion matrix
    headers = [ str(t) for t in range(1,confusion_matrix.shape[0]+1) ]
    print "\t" + '\t'.join(headers)
    
    total = 0
    col_total = list()
    false_negatives = list()
    for i in range(0, confusion_matrix.shape[0]):
        row_total = sum(confusion_matrix[i])
        col_total.append(str(sum(confusion_matrix[:,i])))
        
        #try:
            #fn = confusion_matrix[i][i]/numpy.sum(confusion_matrix[:,i])
            #false_negatives.append(str(round(fn*100,2)) + "%")
        #except Exception,e:
            #print e
            #pass
        
        total += confusion_matrix[i][i]
        if (row_total <> 0):
            row_correct = confusion_matrix[i][i] / row_total
        else:
            row_correct = 0
        values = [ str(int(v)) for v in confusion_matrix[i] ]
        print str(i+1) + "\t" + ('\t').join(values) + "\t(" + str(row_total) + ")\t" + str(round(row_correct*100,2)) + "%"
    
    print "\n\t" + ('\t').join(col_total)
    print "\n\t" + ('\t').join(false_negatives)
    print '\nTotal Correct: ' + str(total) + " / " + str(numpy.sum(confusion_matrix)) + "  (" + str(round(total/numpy.sum(confusion_matrix) * 100, 2)) + "%)"
    print '\n'

 ################## MODEL
 
class Model:
    def __init__(self, nn, notes=''):
        self.network = nn
        self.clearStats()
        self.notes = notes
        
        self.norm_avg = 0
        self.norm_std = 1
        
        # track performance of train, valid sets
        self.train_results = list()
        self.valid_results = list()
        
        self.modelStats = defaultdict(list)
        self.transformer = None
        
    def showResults(self):
        print "Training results: "
        print [ str(round(t*100,2)) + "%" for t in self.train_results ]
        print "Valid results: "
        print [ str(round(t*100,2)) + "%" for t in self.valid_results ]
        # for javascript chart
        print ''.join([ str(idx) + "," + str(t_v[0]) + "," + str(t_v[1]) + "\\n" for idx, t_v in enumerate(zip(self.train_results,self.valid_results)) ])
        
    def clearStats(self):
        self.trainPerformance = list()
        self.validPerformance = list()
        self.testPerformance = list()
        
        self.trainError = list()
        self.validError =list()
        self.testError = list()    
    
    def runModel(self, epochs, trainData, testData,  updateType='', validation=0, early_stopping=-1, batchSize=1, screen_debug=False, noise=0, normalize = True, classification = False, minMax=False, statsBatchInterval=0, statsEpochInterval=0, runsPath='', norm_with_test=False):
        arguments = locals()
        print self.notes
        print '/**************************'
        for arg in arguments.keys():
            if arg <> 'self':
                if ((arg == 'trainData') or (arg == 'testData')):
                    if (arguments[arg]):
                        try:
                            print '*  ' + arg + '.size = ' + str(len(arguments[arg]['x']))
                        except:
                            pass
                else:
                    print '*  ' + arg + ' = ' + str(arguments[arg])
                    
        print '*  self.network.learn_rate = ' + str(self.network.learn_rate)
        print '*  self.network.momentum = ' + str(self.network.momentum)
        print '**************************/\n'
        
        self.clearStats()
        trainData['x'] = trainData['x'].astype('float')
        trainData['t'] = trainData['t'].astype('float')


        # set start and end index points for training and validation
        # if validation = 0 then this uses entire training set and doesn't use a validation set
        train_idx = trainData['x'].shape[0] - validation
        valid_idx = trainData['x'].shape[0]
        
        # setup list of random indices using numpy.random.shuffle
        e = 0
        self.prev_correct = list()
        idx = range(0, trainData['x'].shape[0])
        numpy.random.shuffle(idx)
        
        validData = dict()
        if validation <> 0:
            validData['x'] = trainData['x'][idx[0:validation]]
            validData['t'] = trainData['t'][idx[0:validation]]
        else:
            validData['x'] = None
            validData['t'] = None

        trainData['x'] = trainData['x'][idx[validation:]]
        trainData['t'] = trainData['t'][idx[validation:]]

        def screen_print(text):
            if screen_debug:
                print text
                
        if norm_with_test:
            if self.transformer:
                transform_testData = self.transformer.transform(testData['x'])
                print transform_testData.shape
                print trainData['x'].shape
                scale_use_data = numpy.vstack((transform_testData,trainData['x']))
            else:
                scale_use_data = numpy.vstack((testData['x'],trainData['x']))
        else:
            scale_use_data = trainData['x'][idx[0:train_idx]]
        
        print scale_use_data.shape
        # normalize data
        if normalize:
            # use training data to compute average and standard deviation
            avg = numpy.average(scale_use_data, axis=0)
            std = numpy.std(scale_use_data, axis=0)
            # avoid dividing by nan or inf
            std[numpy.isinf(std)] = 1
            std[numpy.isnan(std)] = 1
            std[std==0] = 1
            
            # now modify all of the trainData and testData with these averages
            trainData['x'] = trainData['x'] - avg
            trainData['x'] = trainData['x']/std
            
            #if (testData.has_key('x')):
                #testData['x'] = testData['x'] - avg
                #testData['x'] = testData['x'] / std
        
            self.norm_avg = avg
            self.norm_std = std
        
        if minMax:
            max_x = numpy.max(scale_use_data, axis=0)
            min_x = numpy.min(scale_use_data, axis=0)
            
            max_x[numpy.where(max_x == min_x)] += 1e-20     # divide by zero
            fractions = (trainData['x'] - min_x) / (max_x - min_x)
            # leave it 0->1 although for sigmoid functions this may mess up derivatives...check out Efficient Backprop
            trainData['x'] = fractions
            # to enforce a range
            #trainData['x'] = fractions * 0.8 + (1-fractions) * 0.2
            
            self.max_x = max_x
            self.min_x = min_x
        
        # track performance of train, valid sets
        self.train_results = list()
        self.valid_results = list()
        
        
        
        # for early stopping
        previous_validation_error = 1e10
        total_train_time = list()

        # to use cudarray
        #trainData['x'] = ca.array(trainData['x'])
        #trainData['t'] = ca.array(trainData['t'])

        train_shuffle_idx = range(0, trainData['x'].shape[0])
        numpy.random.shuffle(train_shuffle_idx)
        use_trainData=dict()
        while e < epochs:
            time_epoch = time.time()
            # store stats over epochs so you can view them
            # save stats to files etc to compare, i.e. output a report you can compare with model hyper-parameters, architecture, error, and performance
            self.clearStats()
            correct = 0
            
            screen_print('Epoch ' + str(e) + '\n')
            # note this will add in validation data to the validation set
            # have to redo the entire model file
            numpy.random.shuffle(train_shuffle_idx)
            temp_data = list()
            temp_test = list()
            print "\n"
            print train_shuffle_idx[0]
            for t_idx in train_shuffle_idx:
                temp_data.append(numpy.ravel(trainData['x'][t_idx]))
                temp_test.append(numpy.ravel(trainData['t'][t_idx]))
            
            use_trainData['x'] = temp_data
            use_trainData['t'] = temp_test

            batch_delta_w = None
            batch_delta_b = None
            
            # run training data
            train_start = time.time()
            confusion = None
            correct_class = 0.
            
            cost = 0

            for n in numpy.arange(0, train_idx, batchSize):
                if screen_debug: 
                    utils.pbar('Training', n+1, train_idx)
                print train_shuffle_idx[0]
                #idx_range = train_shuffle_idx[n:(n+batchSize)]
                #print type(trainData)
                #print type(trainData['x'])
                #print n
                #temp = ca.array(trainData['x'][n:(n+batchSize)])
                x = ca.array(use_trainData['x'][n:(n+batchSize)])
                t = ca.array(use_trainData['t'][n:(n+batchSize)])
                
                if noise <> 0:
                    x += ca.random.normal(0, noise, x.shape)
                
                t0 = time.time()
                if n == 0:
                    print x[0]
                    print t[0]
                    raw_input(':')
                
                delta_w, delta_b, err, output = self.network.train(x, t, e)
                total_train_time.append(time.time() - train_start)
                # confusion matrix
                if (self.network.layers[-1].f == activation.softmax) :
                    if confusion is None:
                        # initialize
                        confusion = numpy.zeros((output.shape[1], output.shape[1]))

                    
                    
                    output_classes = numpy.argmax(output, axis=1)
                    target_classes = numpy.argmax(t, axis=1)
                    for cf_idx,row in enumerate(target_classes):
                        col = output_classes[cf_idx]
                        confusion[col][row] += 1.

                if classification:
                    # note for dropout the output reported above isn't really going to work since we need to rescale on the forward pass, uncomment this if you're using dropout -- refactor
                    output_class = numpy.round(output)      # round the output to either 1 or 0
                    train_class_err = numpy.abs(t - output_class)
                    train_class_err = numpy.dot(train_class_err.T, train_class_err)
                    correct_class += len(output) - list(numpy.array(train_class_err))[0]
                    correct_class = correct_class[0]
                
                self.network.update(updateType, delta_w, delta_b)
#                if (statsBatchInterval <> 0) and (statsEpochInterval <> 0):
#                    if (((n / batchSize) % statsBatchInterval) == 0) and (n <> 0) and (e % statsEpochInterval == 0):
#                        print "\nSNAP! " + str(n/batchSize) + "," + str((n/batchSize) % statsBatchInterval) + "," + str(e % statsEpochInterval) + "\n"
#                        network_weights = [ l.W.tolist() for l in self.network.layers ]
#                        network_bias = [ l.B.tolist() for l in self.network.layers ]

#                        weight_path = runsPath + "/" + str(e) + "/weights/"
#                        error_path = runsPath + "/" + str(e) + "/errors/"
#                        result_path = runsPath + "/" + str(e) + "/results/"
                        
#                        if not os.path.exists(weight_path):
#                            os.makedirs(weight_path)
#                        if not os.path.exists(error_path):
#                            os.makedirs(error_path)
#                        if not os.path.exists(result_path):
#                            os.makedirs(result_path)

                        #fout = gzip.open(weight_path + str(n) + ".net-weights.json.gz", 'w')
#                        fout = open(weight_path + str(n) + ".net-weights.json", 'w')
#                        json.dump(network_weights, fout)
#                        fout.close()
                        
                        #fout = gzip.open(weight_path + str(n) + ".net-bias.json.gz", 'w')
#                        fout = open(weight_path + str(n) + ".net-bias.json", 'w')
#                        json.dump(network_bias, fout)
#                        fout.close()
                        
                        #fout = gzip.open(error_path + str(n) + ".delta_w.json.gz", 'w')
#                        fout = open(error_path + str(n) + ".delta_w.json", 'w')
#                        json.dump([ w.tolist() for w in delta_w], fout)
#                        fout.close()
                        
                        #fout = gzip.open(error_path + str(n) + ".delta_b.json.gz", 'w')
#                        fout = open(error_path + str(n) + ".delta_b.json", 'w')
#                        json.dump([ b.tolist() for b in delta_b], fout)
#                        fout.close()
                        
                        #fout = gzip.open(result_path + str(n) + ".results.json.gz", 'w')
#                        fout = gzip.open(result_path + str(n) + ".results.json", 'w')
#                        json.dump(confusion.tolist(), fout)
#                        fout.close()
                        
                    
                self.trainPerformance.append(time.time() - t0)
                # note this is a performance error, not a network cost function
                cost += ca.sum(self.network.cost(output,t))
                #self.trainError.extend(cost)
            
            #average_train_error = numpy.average(self.trainError)
            average_train_error =  (batchSize * cost) / train_idx
            
            # outputs
            screen_print('\nTraining:')
            screen_print('\tAverage performance: ' + str(numpy.average(self.trainPerformance)))
            screen_print('\tAverage error: ' + str(average_train_error))
            if classification:
                screen_print("\tCorrect class: " + str(round(correct_class/train_idx * 100, 2)) + "%")
            # get an idea of how weights are evolving and 
            # for verifying adaptive learning rates
            avg_gain_W = list()
            avg_W = list()
            for l in self.network.layers:
                avg_W.append(numpy.average(l.W))
                avg_gain_W.append(numpy.average(l.G_W))
            screen_print('\tAverage gain: ' + str(avg_gain_W))
            screen_print('\tAverage weights: ' + str(avg_W))
            if (self.network.layers[-1].f == activation.softmax) :
                screen_print("\tConfusion:\n")
                printConfusionMatrix(confusion)
                self.train_results.append(numpy.trace(confusion) / numpy.sum(confusion))
            elif classification:
                self.train_results.append(correct_class / train_idx)
            
            # run validation data
            confusion = None
            if (train_idx <> valid_idx):
                correct = 0
                correct_class = 0
                for n,i in enumerate(idx[train_idx:valid_idx]):
                    if screen_debug: utils.pbar('Validation', n+1, validation)
                    x = trainData['x'][i]
                    t = trainData['t'][i]
                    
                    t0 = time.time()
                    output = self.network.forward(x)

                    output_layer = self.network.layers[-1]

                    # These all need to be tidied up properly
                    
                    if classification:
                        output_class = numpy.round(output)      # round the output to either 1 or 0
                        train_class_err = t - output_class
                        train_class_err = numpy.dot(train_class_err, train_class_err.T)
                        correct_class += len(train_class_err) - numpy.sum(train_class_err)
                    
                    if (output_layer.f == activation.softmax):
                        # set the selected class to the maximum probability output
                        selected = numpy.ones(output.shape) * (output == output.max())
                        err = t - selected
                        err = t - (output == output.max())
                        err = numpy.dot(err, err.T)
                        if (err == 0):
                            correct += 1.
                        self.validError.append(err)
                        
                        # confusion matrix
                        if confusion is None:
                            # initialize
                            confusion = numpy.zeros((t.shape[1], t.shape[1]))

                        output_classes = numpy.argmax(output, axis=1)
                        target_classes = numpy.array(numpy.argmax(t, axis=1))
                        for cf_idx,row in enumerate(target_classes.T[0]):
                            col = output_classes[cf_idx]
                            confusion[col][row] += 1
                    else:
                        #self.validError.append(numpy.linalg.norm(output - t) / numpy.linalg.norm(output))
                        self.validError.append(self.network.cost(output, t))
                        
                    self.validPerformance.append(time.time() - t0)
                    
                screen_print('\nValidation:')
                screen_print('\n\tAverage performance: ' + str(numpy.average(self.validPerformance)))
                screen_print('\tAverage error: ' + str(numpy.average(self.validError)))
                if classification:
                    screen_print("\tCorrect class: " + str(round(correct_class/n * 100, 2)) + "%")
                    
                if (output_layer.f == activation.softmax):
                    screen_print("\tCorrect: " + str(round(correct/n * 100, 2)) + "%")
                    screen_print("\tConfusion:\n ")
                    printConfusionMatrix(confusion)
                    self.valid_results.append(numpy.trace(confusion) / numpy.sum(confusion))
                elif classification:
                    self.valid_results.append(correct_class / n)
                
            # run test data
            #if testData.has_key('x'):
                #for i,data in enumerate(testData['x']):
                    #if screen_debug: utils.pbar('Test', i+1, testData['x'].size)
                    #x = data
                    #t = testData['t'][i]
                    
                    #t0 = time.time()
                    #output = self.network.forward(x)
                    #self.testPerformance.append(time.time() - t0)
                    ## use the networks cost function
                    #self.testError.append(self.network.cost(output, t))
                
                #screen_print('\nTest:')
                #screen_print('\n\tAverage performance: ' + str(numpy.average(self.testPerformance)))
                #screen_print('\tAverage error: ' + str(numpy.average(self.testError)))
            
            
            # early_stopping here means something different than typically used, it's basically  the buffer we have before stopping when the error increases
            # instead of the level of error improvement that forces early stopping
            if (train_idx <> valid_idx):
                if (previous_validation_error < (numpy.average(self.validError) - early_stopping)) and (early_stopping <> -1):
                    print "Early stop, breaking."
                    break
                previous_validation_error = numpy.average(self.validError)
                
            screen_print("Stat snapshots: " + str(len(self.modelStats[e])))
            e += 1
            screen_print('Epoch run time ' + str(time.time() - time_epoch))
            screen_print('==============================================')
        
        print "Average training time over all samples per epoch: " + str(numpy.average(total_train_time))
