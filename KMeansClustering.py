import numpy as np
import scipy.stats as stats
import operator
import functools
import sys
import random
import matplotlib.pyplot as plt

infile = sys.argv[1]
k = int(sys.argv[2]) # number of clusters to group data in

data = np.loadtxt(infile) # load infile text file as numpy ndarray

n = data.shape[0] # number of data points
d = data.shape[1] # degree of data points

centroids = data[np.random.randint(0, n, k)] # pick k random datapoints from data

# calculate euclidean distance
def dist(a, b):
    return np.linalg.norm(a-b)

# create ndarray of zeros of the same length and datatype as data
assigned = np.zeros((n,1), dtype = data.dtype)

i = 0
while(i < 20):

    assignedPlus = np.append(assigned, data, axis = 1) # new array w/ assigned as first column and data as the rest

    clusters = [ [] for x in range(k)]  # new list of lists to hold clusters

    # for each data point find the closest centroid
    for x in range(n):
        distances = []
        for y in range(m):
            distances.append(dist(data[x], centroids[y]))
        assignedPlus[x][0] = np.argmin(distances)

    newCentroids = np.zeros((k, d), dtype = centroids.dtype) # ndarray same dimensions as centroids

    # create clusters and calculate their new centroids
    for x in range(k):
        clusters[x] = assignedPlus[assignedPlus[:,0] == x]
        clusters[x] = clusters[x][:, 1:]
        newCentroids[x] = np.mean(clusters[x][:,:], axis =0)

    # calculate the total distance that all of the centroids moved
    sigma = sum(dist(centroids[x], newCentroids[x]) for x in range(k))
    np.copyto(centroids, newCentroids) #update centroids

    if(sigma < 0.001): break

    i += 1

# if data is 2 dimensional plot it
if(d == 2):
    for cluster in clusters:
        plt.plot(cluster[:, 0], cluster[:, 1], 'o')
    for centroid in centroids:
        plt.plot(centroid[0], centroid[1], 'v', markersize = 20)
    plt.axis('equal')
    plt.show()
