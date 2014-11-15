#! /usr/bin/env python
# Logistic regression
import numpy as np
from numpy import linalg as LA
from scipy.special import expit
import scipy
import argparse
import sys
import time
import math
import logging
import itertools
import math
import os
import os.path

def timeit(msg=None):
    # a decorator for time calculating
    if msg:
        logging.info('{0}{1}'.format(msg, time.time() - timeit.start))
    else:
        timeit.start = time.time()

def read_config():
    # read and parse configuration from 'DATA.TXT'
    f = open('DATA.TXT')
    lines = f.readlines()
    d = {}
    for l in lines:
        try:
            k, v = l.strip().split('=')
            d[k] = v
        except:
            pass
    return d

def L2norm(vector):
    s = math.sqrt(reduce(lambda x, y: x + y * y, vector))
    return map(lambda x: x / s, vector)

def read_train_data(filename):
    # read date, returns the data matrix X and class value y
    timeit()
    with open(filename) as f:
        raw_data = map(lambda l: l.strip().split(), f.readlines())
    data = {}
    for row in xrange(len(raw_data)):
        sign = int(raw_data[row][0])
        query = int(raw_data[row][1].split(':')[1])
        row_data = L2norm([float(x.split(':')[1]) for x in raw_data[row][2:-3]])
        if query not in data:
            data[query] = [[], []]
        data[query][sign].append(row_data)
    instances = []
    target = []
    for query in data:
        for irrelavant in data[query][0]:
            for relavant in data[query][1]:
                # add bias
                # positive instance
                instances.append([1] + map(lambda x, y: x - y, relavant, irrelavant))
                target.append(1)
                # negative instance
                instances.append([1] + map(lambda x, y: y - x, relavant, irrelavant))
                target.append(0)
    timeit('Train data reading time:')
    return np.array(instances), np.array(target)

def read_test_data(filename):
    # read date, returns the data matrix X and class value y
    timeit()
    with open(filename) as f:
        raw_data = map(lambda l: l.strip().split(), f.readlines())
    data = []
    for row in xrange(len(raw_data)):
        sign = int(raw_data[row][0])
        query = int(raw_data[row][1].split(':')[1])
        data.append([1] + L2norm([float(x.split(':')[1]) for x in raw_data[row][2:-3]]))
    timeit('Test data reading time:')
    return np.array(data)

def train(X, y, eps=None, threshold=0.005, c=1):
    # training process, return trained weights
    if eps is None:
        eps = 0.00005
    logging.info('Learning rate {0}'.format(eps))
    # initial weight is all zero
    W = np.zeros(X.shape[1])
    timeit()
    delta = None
    i = 0
    while delta is None or LA.norm(delta) > threshold:
        i += 1
        if i % 1000 == 0:
            logging.info('iteration {0}'.format(i))
        # sigmoid[i] = sigmoid(w.T * x[i])
        sigmoid = expit(X.dot(W))
        # distance[i] = y[i] - sigmoid(w.T * x[i])
        distance = y - sigmoid
        # Z[i,j] = partial L_i(w) / partial w[j]
        Z = X * distance[:, np.newaxis]
        # summation
        gradient = Z.sum(axis=0)
        # regularization
        regular = c * W
        regular[0] = 0
        # update weight
        delta = eps * (gradient - regular)
        W += delta
    timeit('Training time:')
    logging.info('Weight = {0}'.format(W))
    return W

def predict(X, W):
    # predict process, return predictions
    score = X.dot(W)
    return score

def main():
    d = read_config()
    if len(sys.argv) > 1:
        d['c'] = sys.argv[1]
    logging.basicConfig(filename='run{0}.log'.format(d['c']),level=logging.DEBUG)
    tX, tY = read_train_data(d['train'])
    logging.info('Train shape = {0}'.format(tX.shape))
    pX = read_test_data(d['test'])
    logging.info('Test shape = {0}'.format(pX.shape))
    W = train(tX, tY, c=float(d['c']))
    yHat = predict(pX, W)
    for hat in yHat:
        print hat

if __name__ == '__main__':
    main()