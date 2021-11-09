''' Hierarchical clustering '''
# agglomerative approach (top down) - many clusters to few
# diffusive (bottom-up) - single cluster to many

# Doesn't utilise centroids, unlike kmeans
# & doesn't make any assumptions about the data or clusters
# no predict or centroid_

import numpy as np
from sklearn.cluster import AgglomerativeClustering

data = np.array([
  [5.7, 5.5, 1.4, 0.15],
  [4.6, 3.7 , 7.4, 0.16],
  [4.7, 3.9, 6.3, 0.66],
  [4.9, 3.4, 2.5, 0.72],
  [5.2 , 4.1, 3.4, 0.32]])

from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=3)
model.fit(data)

# cluster assignments
print(model.labels_)
