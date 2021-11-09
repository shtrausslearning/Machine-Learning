import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# similarity of values within array
data = np.array([
  [ 1.8,  1.3],
  [ 2.6,  6.6],
  [-1.5, -1.4],
  [ 0.2 , -6.2]])

similarity = cosine_similarity(data)
print(f'{(similarity)}')

# similarity b/w two arrays
data1 = np.array([
  [ 1.8,  1.3],
  [ 2.6,  6.6],
  [-1.5, -1.4],
  [ 0.2 , -6.2]])
data2 = np.array([
  [ 7.7,  0.4],
  [ 6.2, 1.85],
  [8.0,  1.12]])
similarity = cosine_similarity(data1, data2)
print(f'{(similarity)}')
