#!/usr/bin/env python2

from collections import defaultdict
import time
import numpy as np

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
       
        try: 
            fn = 1 -(confusion_matrix[i][i]/np.sum(confusion_matrix[:,i]))
            false_negatives.append(str(round(fn*100,2)) + "%")
        except Exception,e:
            print e
            pass
        
        total += confusion_matrix[i][i]
        if (row_total <> 0):
            row_correct = confusion_matrix[i][i] / row_total
        else:
            row_correct = 0
        values = [ str(int(v)) for v in confusion_matrix[i] ]
        print str(i+1) + "\t" + ('\t').join(values) + "\t(" + str(row_total) + ")\t" + str(round(row_correct*100,2)) + "%"
    
    print "\n\t" + ('\t').join(col_total)
    print "\n\t" + ('\t').join(false_negatives)
    print '\nTotal Correct: ' + str(total) + " / " + str(np.sum(confusion_matrix)) + "  (" + str(round(total/np.sum(confusion_matrix) * 100, 2)) + "%)"
    print '\n'


class Trainer:
    # use parameters to define specific parameters for a trainer
    def __init__(self, parameters):
        self.parameters = parameters

    def snapshot(self):
        pass


class NNTrainer:
    def __init__(self, parameters):
        print parameters
        if not parameters.has_key('epochs'):
            parameters['epochs'] = 1
        if not parameters.has_key('updateType'):
            parameters['updateType'] = 'vanilla'
        if not parameters.has_key('batchSize'):
            parameters['batchSize'] = -1
        
        self.parameters = parameters
        self._postUpdate = self.postUpdate
        self.statistics = defaultdict(list)

    def train(self, network, data):
        if not data.transformed:
            data.transform()
        data.reset()
        e = 0
        while (e < self.parameters['epochs']):
            data.shuffle()
            for i in range(0, data.length, self.parameters['batchSize']):
                values, targets = data.getbatch(self.parameters['batchSize'])
                if values is not None:
                    delta_w, delta_b, err, output = network.train(values, targets)
                    network.update(self.parameters['updateType'], delta_w, delta_b) 
                self._postUpdate(delta_w, delta_b, err, output, values, targets)
            e += 1
            data.reset()    # reset position

    def postUpdate(self, delta_w, delta_b, error, output, values, targets):
        pass
    
    def setPostUpdate(self, f):
        self._postTrain = f

    def confusion(self, network, data):
        if not data.transformed:
            data.transform()
        x, y = data.getbatch(self.parameters['batchSize'])
        confusion_matrix = np.zeros((y.shape[1], y.shape[1]))
        idx = 0
        while x is not None:
            output = network.forward(x, train=False)
            output_class = np.argmax(output, axis=1)
            target_class = np.argmax(y, axis=1)
            for o, t in zip(output_class, target_class):
                confusion_matrix[o][t] += 1
            idx += 1
            x, y = data.getbatch(self.parameters['batchSize'])
        return confusion_matrix

if __name__ == "__main__":
    values = np.matrix(range(0,100)).T
    targets = np.matrix(range(0,100)).T
    test = dataset(values, targets)
    test.shuffle()
    print test.getbatch(10)
        
        
        
