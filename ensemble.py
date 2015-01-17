#!/usr/bin/env python2

import numpy as np
from trainer import NNTrainer
import trainer
from dataset import dataset
from sklearn import linear_model

class Ensemble:
    def __init__(self, base_net, models, model_memory=0):
        # for now model memory isn't implemented, i.e. when selecting
        # parameters I'm using single point sampling.  Model paramter samples don't
        # interact

        # note base_net is NOT used as part of the ensemble so don't train it
        self.model_memory = float(model_memory)
        self.num_models = int(models)
        self.models = list()
        self.base_net = base_net

    def forward(self, data, train=False):
        # the train parameter shouldn't be set to True ever
        # this is just to allow code which call network.forward() to also call ensemble.forward()
        # there's definitely a more elegant way to do this
        output = None
        for idx,m in enumerate(self.models):
            if output is None:
                output = m.forward(data, train) * self.model_weights[idx]
            else:
                output += m.forward(data, train) * self.model_weights[idx]

        return output

    def generic(self, data, trainer_params, nambla=0.5, lasso_alpha=0.1):
        # follow generic ensemble generation
        # data is a dataset
        net_train = NNTrainer(trainer_params)

        data.transform()

        for i in range(0, self.num_models):
            data.reset()
            data.shuffle()
            # a bit inefficient, copying back and forth here
            sub_x, sub_y = data.getbatch(data.length * nambla)
            sub_x = np.array(sub_x)
            sub_y = np.array(sub_y)
            sub_sample = dataset(sub_x, sub_y, transformer = data.transformer, backend_type = data.backlib)
            sub_sample.transformed = True
            net = self.base_net.clone()
            print "Training network " + str(i)
            # note that if we had memory_model we would need to augment
            # each nets output by an accumulated prediction
            net_train.train(net, sub_sample)
            #c = net_train.confusion(net, sub_sample)            
            #trainer.printConfusionMatrix(c)
            self.models.append(net)

        data.reset()
        print "Trained " + str(len(self.models)) + " models"

        self.model_weights = self.base_net.back.ones((self.num_models,1)) * (1 * (self.num_models** -1))
        # the below coefficient search isn't implemented correctly but...
        # to do it properly for a classificaiton problem you would need an NxMxK
        # matrix and pick coefficients to minimize loss of a linear combination
        # of the outputs
        # tried doing this by converting each softmax output to a single class but I may
        # be doing it incorrectly.  For now see above, simple average of outputs is used
        #clf = linear_model.Lasso(lasso_alpha)
        # setup matrix of losses for all models and all training data
        #for i in range(0, self.num_models):
        #    values, targets = data.getbatch(1000)
        #    while values is not None:
        #        output = self.models[i].forward(values, train=False)
        #        output_class = np.argmax(output, axis=1)
        #        values, targets = data.getbatch(1000)
        #        try:
        #            output_matrix = np.hstack((output_matrix,output_class))
        #        except UnboundLocalError:
        #            output_matrix = output_class
        # 
        #clf.fit(output_matrix, data.np_y.T)
        #print clf.intercept_
        #print clf.coef_
        #print clf.sparse_coef_
        
        
