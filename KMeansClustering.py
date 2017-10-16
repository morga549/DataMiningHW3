import numpy as np
import scipy.stats as stats
import operator
import functools
import sys
import random

infile = sys.argv[1]
k = int(sys.argv[2])

data = np.loadtxt(infile)
print type(data.shape[0])
print k

n = data.shape[0]
d = data.shape[1]

centroids = data[np.random.randint(0, n, k)]

print(centroids)
