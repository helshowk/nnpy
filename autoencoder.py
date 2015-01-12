#!/usr/bin/env python2

import numpy

import neural
import activation
import cost

fin = open('Kaggle/Forest/train.csv', 'r')
lines = fin.readlines()
fin.close()

# ignore first line as it has headers
trainData = dict()
x_data = list()
t_data = list()

for l in lines[1:]:
	l = l.rstrip('\n')
	temp = l.split(',')
	temp = [int(x) for x in temp]
	
	length = len(temp)
	target = [ 0,0,0,0,0,0,0 ]
	target[int(temp[(length-1)])-1] = 1
	temp = temp[1:]
	feat_vec = temp[0:54]	
	
	# modify feature vector, collapse wild_type and soil_type
	wild_type = feat_vec[10:14]
	soil_type = feat_vec[14:54]
	#feat_vec = feat_vec[0:10]
	#feat_vec.append(wild_type.index(1))
	#feat_vec.append(soil_type.index(1))
	
	# remove id and target from feature vector
	x_data.append(feat_vec)
	t_data.append(target)


trainData['x'] = numpy.matrix(x_data)
trainData['t'] = numpy.matrix(t_data)

# normalize first columns
avg = numpy.average(trainData['x'], axis=0)
std = numpy.std(trainData['x'], axis=0)

# uncomment if you aren't collapsing binary strings
avg[..., 10:54] = 0
std[..., 10:54] = 1
trainData['x'] = trainData['x'] - avg
trainData['x'] = trainData['x'] / std

testData = dict()

network = neural.NN(cost.quadratic, cost.dquadratic)
network.addLayer(neural.Layer(54,25,activation.tanh, activation.d_tanh))
network.addLayer(neural.Layer(25,54,activation.identity, activation.one))

#network.momentum = 0.75
#network.learn_rate = 0.25

m = neural.Model(network, notes='Forest')
m.runModel(20, trainData, testData, validation=3000, batchSize=50, screen_debug=True, plot_error = False)

print "Saving model..."
fout = open(sys.argv[1], 'wb')
pickle.dump(m, fout)
fout.close()
