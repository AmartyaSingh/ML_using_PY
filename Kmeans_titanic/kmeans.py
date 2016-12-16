import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[2,3],[3,4],[1,2],[6.8,6],[7,7],[10,8]])
#plt.scatter(X[:,0], X[:,1], s=150) #X[:,0] -> axis zero, i.e. all x value, X[:,0] -> all y values
#plt.show()
clf = KMeans(n_clusters=2)
clf.fit(X)
centroids = clf.cluster_centers_
labels = clf.labels_

colors = ["g.", "r.", "c.", "b.", "k.", "o."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 25)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=150, linewidth=5)
plt.show()
