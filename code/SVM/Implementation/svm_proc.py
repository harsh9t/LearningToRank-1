#! /usr/bin/env python
# Logistic regression
import numpy as np
from numpy import linalg as LA
import argparse
import sys
import time
import math
import logging
import itertools
import math
import os
import os.path

def L2norm(vector):
    s = math.sqrt(reduce(lambda x, y: x + y * y, vector))
    return map(lambda x: x / s, vector)

def transform_train(infile, outfile):
    with open(infile) as f:
        raw_data = map(lambda l: l.strip().split(), f.readlines())
    data = {}
    for row in xrange(len(raw_data)):
        sign = int(raw_data[row][0])
        query = int(raw_data[row][1].split(':')[1])
        row_data = L2norm([float(x.split(':')[1]) for x in raw_data[row][2:-3]])
        if query not in data:
            data[query] = [[], []]
        data[query][sign].append(row_data)
    with open(outfile, 'w') as f:
        for query in data:
            for irrelavant in data[query][0]:
                for relavant in data[query][1]:
                    row = map(lambda x, y: x - y, relavant, irrelavant)
                    row = ['%d:%f' % (1+x[0],x[1]) for x in enumerate(row)]
                    line = ' '.join(['+1'] + row)
                    f.write('%s\n' % line)

def transform_test(infile, outfile):
    with open(infile) as f:
        raw_data = map(lambda l: l.strip().split(), f.readlines())
    with open(outfile, 'w') as g:
        for raw in raw_data:
            transformed = L2norm([float(x.split(':')[1]) for x in raw[2:-3]])
            transformed = ['%d:%f' % (1+x[0],x[1]) for x in enumerate(transformed)]
            line = ' '.join(raw[:2] + transformed + raw[-3:])
            g.write('%s\n' % line)

def main():
    transform_train(sys.argv[1], sys.argv[2])
    transform_test(sys.argv[3], sys.argv[4])

if __name__ == '__main__':
    main()