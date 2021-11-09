''' DBSCAN clustering '''
# more scalable than meanshift
# doesn't make assumption that data groups are blobs
# chooses automatically number of clusters
# uses region/scatter density to group
# makes no assumption about shape of cluster

import numpy as np
from sklearn.cluster import DBSCAN

data = np.array([
  [5.7, 5.5, 1.4, 0.15],
  [4.6, 3.7 , 7.4, 0.16],
  [4.7, 3.9, 6.3, 0.66],
  [4.9, 3.4, 2.5, 0.72],
  [5.2 , 4.1, 3.4, 0.32],
  [5.9, 7.4, 1.5, 0.72],
  [6.2 , 5.1, 2.4, 1.33],
  [4.5, 3.5 , 7.3, 0.13],
  [4.7, 3.2, 6.6, 0.6],
  ])

# DBSCAN moedl
model = DBSCAN(eps=1.2, min_samples=1)
model.fit(data)

print(model.labels_) # cluster assignments
print(model.core_sample_indices_) # sample index

num_core_samples = len(model.core_sample_indices_)
print(f'core samples: {num_core_samples}')
