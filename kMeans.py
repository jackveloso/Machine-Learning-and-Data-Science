import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from sklearn.datasets import make_blobs


def kMeans(data, k=2):
    '''
    Return the kMeans centroids and clusters.
    '''
    centroids = data[:k]
    oldcentroids = []
    while True:
        oldcentroids = centroids[:]
        clusters = [[] for _ in centroids]
        # Update clusters
        for item in data:
            distances_to_centroids = [np.dot(centroid-item, centroid-item)
                                      for centroid in centroids]
            ind_of_closest_centroid = np.argmin(distances_to_centroids)
            clusters[ind_of_closest_centroid].append(item)
        # Update centroids
        centroids = np.array([np.array(cluster).mean(axis=0)
                              for cluster in clusters])
        if (centroids == oldcentroids).all():
            return centroids, clusters


np.random.seed(1234)
X, y = make_blobs(n_samples=200, centers=3)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("3 Random Blops")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()
centroids, clusters = kMeans(X, k=3)
colors = iter(cm.rainbow(np.linspace(0, 1, len(centroids))))
fig = plt.figure(figsize=(8, 6))
for centroid, cluster in zip(centroids, clusters):
    color = next(colors)
    plt.scatter(np.array(cluster)[:, 0], np.array(cluster)[:, 1], c=color)
    plt.scatter(centroid[0], centroid[1], s=75, c=color, marker='s', linewidths=5)
plt.title("The 3 blops with each center")
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.show()
