import numpy as np
import scipy.stats as stats
import operator
import functools
import sys
import random
import matplotlib.pyplot as plt

infile = sys.argv[1]
k = int(sys.argv[2])

data = np.loadtxt(infile)

n = data.shape[0]
d = data.shape[1]

centroids = data[np.random.randint(0, n, k)]
m = centroids.shape[0]

def dist(a, b):
    return np.linalg.norm(a-b)

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
        clusters[x] = clusters[x][:, 1:]
        newCentroids[x] = np.mean(clusters[x][:,:], axis =0)

    sigma = sum(dist(centroids[x], newCentroids[x]) for x in range(k))
    np.copyto(centroids, newCentroids)

    if(sigma < 0.001): break

    i += 1

if(d == 2):
    for cluster in clusters:
        plt.plot(cluster[:, 0], cluster[:, 1], 'o')
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], 'v', markersize = 20)
    plt.axis('equal')
    plt.show()
