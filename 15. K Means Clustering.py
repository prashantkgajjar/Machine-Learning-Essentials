# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# ------------ K Means Clustering -------------


'''
********** Intuition ***********
1. Choose K number of clusters
2. Select random K points, i.e. (centroids) - (not necessarily from dataset)
3. Assign each centroids (K) to the nearest datapoints. (nearest can be calculated from
   with the help of Euclidean Distances)
4. Compute and place the new centroid to each cluster
5. Reassign each datapoints to the closest centroid.
6. If any reassignment took place, go to STEP4, otherwise go to FIN.

* The selection of random K points will eventually result in falsly selecting
  the incorrect centroids. These will eventually result in incorrect cluster
  formation.
  
  In order to avoid this, there is a 'K Means ++ Cluster Algorithm' which is better than 
  'K Means Clusters'
  
'''
'''
*********** Selecting right number of K (clusters) *************
 Visualize the WCSS vs Number of clusters, and select the cluster through
 elbow method.
 
 '''
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 

