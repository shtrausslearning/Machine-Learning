''' Standard KMeans '''
# dataset consists of spherical clusters
# uses centroids 

import numpy as np
from sklearn.cluster import KMeans

data = np.array([
  [5.7, 5.5, 1.4, 0.15],
  [4.6, 3.7 , 7.4, 0.16],
  [4.7, 3.9, 6.3, 0.66],
  [4.9, 3.4, 2.5, 0.72],
  [5.2 , 4.1, 3.4, 0.32]])

# KMeans Model /w n_clusters
model = KMeans(n_clusters=3)
model.fit(data)

print(model.labels_) # cluster assignments
print(model.cluster_centers_) # cluster centre

# New Observation
new_obs = np.array([
  [5.3, 3.6, 1.7, 1.6],
  [6.9, 3.2, 5.3, 0.1]])
# predict clusters
print(model.predict(new_obs))

''' When working with large dataset '''
from sklearn.cluster import MiniBatchKMeans

# KMeans Model /w n_clusters + number of batches
model = MiniBatchKMeans(n_clusters=3, batch_size=5)
model.fit(data)

print(model.labels_) # cluster assignments
print(model.cluster_centers_) # cluster centre

new_obs = np.array([
  [5.3, 3.6, 1.7, 1.6],
  [6.9, 3.2, 5.3, 0.1]])

# predict clusters
print(model.predict(new_obs))
