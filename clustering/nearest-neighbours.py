data = np.array([
  [5.7, 5.5, 1.4, 0.15],
  [4.6, 3.7 , 7.4, 0.16],
  [4.7, 3.9, 6.3, 0.66],
  [4.9, 3.4, 2.5, 0.72],
  [5.2 , 4.1, 3.4, 0.32]])

import numpy as np
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors() # k=5 default
model.fit(data)

# singular array observation
new_obs = np.array([[5.5 , 1.5, 0.6, 4.3]])

# return both distances & neighbours
dists, nbrs = model.kneighbors(new_obs)

print(nbrs) # nearest neighbors indexes
print(dists) # nearest neighbor distances

# if we wanted to return only neighbours
only_nbrs = model.kneighbors(new_obs,return_distance=False)
print(only_nbrs)

# If we wanted to change the num. nearest neighbours
model = NearestNeighbors(n_neighbors=2)
model.fit(data)

# multidimensional array observation
new_obs = np.array([[5.0 , 7.5, 6.6, 1.2],
                    [0.1, 5.2, 7.5, 2.1]])
dists, nbrs = model.kneighbors(new_obs)

print(nbrs) # nearest neighbors indexes
print(dists) # nearest neighbor distances
