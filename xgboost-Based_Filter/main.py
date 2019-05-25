#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
import random

def Test():
	data = np.loadtxt('./dermatology.data', delimiter=',',
	        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
	sz = data.shape

	train = data[:int(sz[0] * 0.7), :]
	test = data[int(sz[0] * 0.7):, :]

	train_X = train[:, :33]
	train_Y = train[:, 34]

	test_X = test[:, :33]
	test_Y = test[:, 34]

	xg_train = xgb.DMatrix(train_X, label=train_Y)
	xg_test = xgb.DMatrix(test_X, label=test_Y)
	# setup parameters for xgboost
	param = {}
	# use softmax multi-class classification
	param['objective'] = 'multi:softmax'
	# scale weight of positive examples
	param['eta'] = 0.1
	param['max_depth'] = 6
	param['silent'] = 1
	param['nthread'] = 4
	param['num_class'] = 6

	watchlist = [(xg_train, 'train'), (xg_test, 'test')]
	num_round = 5
	bst = xgb.train(param, xg_train, num_round, watchlist)
	# get prediction
	pred = bst.predict(xg_test)
	error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
	print('Test error using softmax = {}'.format(error_rate))

	# do the same thing again, but output probabilities
	param['objective'] = 'multi:softprob'
	bst = xgb.train(param, xg_train, num_round, watchlist)
	# Note: this convention has been changed since xgboost-unity
	# get prediction, this is in 1D array, need reshape to (ndata, nclass)
	pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
	pred_label = np.argmax(pred_prob, axis=1)
	error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
	print('Test error using softprob = {}'.format(error_rate))


def LiveXGBoostAnalysis():
	termsArray  = []
	docMatrix   = np.zeros((2225, 9636), dtype=int)
	classMatrix = np.zeros((2225, 1), dtype=int)


    #we build the matrix
    #feature1   feature2   ...  featureN   class
    #   1          0       ...      10       4
    #   0          1       ...      3        2
    #   ...
	for line in open("bbc.terms"):
		termsArray.append(line)

	for line in open("bbc.classes"):
		sections = line.strip('\n').split(" ")
		classMatrix[int(sections[0])] = int(sections[1])

	for line in open("bbc.mtx"):
		sections = line.strip('\n').split(" ")
		docMatrix[int(sections[1]) - 1][int(sections[0]) - 1] = int(float(sections[2]))

	for index in range(2225):
	    docMatrix[index][9635] = classMatrix[index]


	sz = docMatrix.shape

	for index in range(2225):
		rowA = random.randint(0,2224)
		rowB = random.randint(0,2224)
		print("exchange : %d, %d\n" % (rowA, rowB))
		docMatrix[[rowA, rowB], :] = docMatrix[[rowB, rowA], :]


	#we train the matrix using xgboost
	train = docMatrix[:int(sz[0] * 0.7), :]
	test = docMatrix[int(sz[0] * 0.7):, :]

	train_X = train[:, :9634]
	train_Y = train[:, 9635]

	test_X = test[:, :9634]
	test_Y = test[:, 9635]

	xg_train = xgb.DMatrix(train_X, label=train_Y)
	xg_test = xgb.DMatrix(test_X, label=test_Y)
	param = {}
	param['objective'] = 'multi:softmax'
	param['eta'] = 0.1
	param['max_depth'] = 6
	param['silent'] = 1
	param['nthread'] = 4
	param['num_class'] = 5
	watchlist = [(xg_train, 'train'), (xg_test, 'test')]
	num_round = 50


	#method one
	bst = xgb.train(param, xg_train, num_round, watchlist)
	pred = bst.predict(xg_test)
	print(pred)
	error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
	print('Test error using softmax = {}'.format(error_rate))


	#method two
	param['objective'] = 'multi:softprob'
	bst = xgb.train(param, xg_train, num_round, watchlist)
	pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 5)
	print(pred_prob)
	pred_label = np.argmax(pred_prob, axis=1)
	error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
	print('Test error using softprob = {}'.format(error_rate))

if __name__ == '__main__':
	#Test()
	LiveXGBoostAnalysis()
