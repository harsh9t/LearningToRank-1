#! /usr/bin/env python
# Logistic regression
import numpy as np
from numpy import linalg as LA
from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import expit
import scipy
import argparse
import sys
import time
import math
import logging
import itertools
import math

KLASSES = 17
COL = 14602

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

def read_data(filename):
    # read date, returns the data matrix X and class value y
    timeit()
    with open(filename) as f:
        raw_data = map(lambda l: l.strip().split(), f.readlines())
    data = []
    i = []
    j = []
    y = []
    for row in xrange(len(raw_data)):
        y.append(int(raw_data[row][0]))
        # add bias x[i,0] = 1
        data.append(1)
        i.append(row)
        j.append(0)
        for col in xrange(1, len(raw_data[row])):
            jj, dd = raw_data[row][col].split(':')
            data.append(float(dd))
            i.append(row)
            j.append(int(jj))
    rows = len(raw_data)
    columns = COL
    X = csr_matrix((data, (i, j)), shape=(rows, columns), dtype=float)
    y = np.array(y)
    timeit('Read:')
    return X, y

def train(X, y, eps=None, threshold=0.01, c=1):
    # training process, return trained weights
    if eps is None:
        eps = min(0.01, 0.01 / c)
        logging.info('eps {0}'.format(eps))
    # initial weight is all zero
    W = np.zeros((KLASSES, X.shape[1]))
    timeit()
    for klass in xrange(KLASSES):
        delta = None
        i = 0
        while delta is None or LA.norm(delta) > threshold:
            i += 1
            if i % 1000 == 0:
                logging.info('iteration {0}'.format(i))
            # sigmoid[i] = sigmoid(w.T * x[i])
            sigmoid = expit(X.dot(W[klass]))
            # distance[i] = y[i] - sigmoid(w.T * x[i])
            distance = (y == (klass + 1)).astype(float) - sigmoid
            # Z[i,j] = partial L_i(w) / partial w[j]
            Z = X.copy()
            Z.data *= distance.repeat(np.diff(Z.indptr))
            # summation
            gradient = Z.sum(axis=0).A1
            # regularization
            regular = c * W[klass]
            regular[0] = 0
            # update weight
            delta = eps * (gradient - regular)
            W[klass] += delta
        logging.info('Class {0} trained'.format(klass + 1))
    timeit('Training time:')
    return W

def predict(X, W):
    # predict process, return predictions
    # one-vs-rest method
    score = X.dot(W.T)
    y = np.argmax(score, axis=1)
    y += 1
    return y

def main():
    d = read_config()
    if len(sys.argv) > 1:
        d['c'] = sys.argv[1]
    logging.basicConfig(filename='run{0}.log'.format(d['c']),level=logging.DEBUG)
    rX, ry = read_data(d['train'])
    W = train(rX, ry, c=float(d['c']))
    tX, ty = read_data(d['test'])
    y = predict(tX, W)
    for k, v in itertools.izip(y, ty):
        print '{0} {1}'.format(k, v)

if __name__ == '__main__':
    main()