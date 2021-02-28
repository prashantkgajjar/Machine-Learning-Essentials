# Hierarchical Clustering

'''
1. Agglomerative Hierarchical Clustering (Many single clusters to one single cluster)
    1. Make each data point as a single point Cluster.
    2. Take the two closest datapoints, and make them one cluster.
    3. Take to closest clusters, and make them one cluster.
    4. Repeat STEP 3 unitll there is only one cluster.
'''

'''
******** Distance between two clusters:*******
    Option 1: Measure Distance between two closest points
    Option 2: Measure Distance between two farthest points
    Option 3: Take the average of the all points of clusters
    Option 4: Measure the distance between two centroids of clusters.
'''

'''
******** Dendograpms ***********
    It is a diagram that shows the Hierarchical relationship between two clusters or 
    data points
    To optimize the clustering, selecting the Dendograms with the Highest Distance is
    considered as a optimal limit of number of cluster.
'''



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

'''
From the above plot, it can be understood that the length(line)
with highest gap between the clusters is the first line orangge one.
From that line, 5 others such line are passsing, so we can conclude that there are 5 such
clusters.
'''


# Training the Hierarchical Clustering model on the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


'''
From the cluster, it can be concluded that:::
    Cluster 1: Red Color -- They spend less, despite earning high
    Cluster 2: Blue Color -- Average customer, those who has average income & average spends
    Cluster 3: Green Color -- They earn more, spend more. Perfect to target for advertisements
    Cluster 4: Light Blue Colo -- They earn less, but mostly spends high at the mall.
    cLUSTER 5: Pink Color -- Thay ean less & spends less
'''
