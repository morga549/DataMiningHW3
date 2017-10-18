import numpy as np
import scipy.stats as stats
import operator
import functools
import sys
import random
from collections import defaultdict
import

infile = sys.argv[1]
k = int(sys.argv[2])

data = np.loadtxt(infile)

n = data.shape[0]
d = data.shape[1]

centroids = data[np.random.randint(0, n, k)]
m = centroids.shape[0]

def dist(a, b):
    return np.sqrt(np.sum((a-b)**2))

assigned = np.zeros((n,1), dtype = data.dtype)

i = 0
while(i < 20):

    assignedPlus = np.append(assigned, data, axis = 1)

    clusters = [ [] for x in range(k)]

    for x in range(n):
        distances = []
        for y in range(m):
            distances.append(dist(data[x], centroids[y]))
        assignedPlus[x][0] = np.argmin(distances)

    newCentroids = np.zeros((k, d), dtype = centroids.dtype)

    for x in range(k):
        clusters[x] = assignedPlus[assignedPlus[:,0] == x]
        newCentroids[x] = np.mean(clusters[x][:, 1:], axis =0)

    sigma = sum(dist(centroids[x], newCentroids[x]) for x in range(k))
    centroids = newCentroids
    print(sigma)
    if(sigma < 0.001): break

    i += 1
