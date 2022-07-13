
## Clustering Approaches 

### 1 | Array Similarity

#### Similarity of values within array

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

data = np.array([
  [ 1.8,  1.3],
  [ 2.6,  6.6],
  [-1.5, -1.4],
  [ 0.2 , -6.2]])

similarity = cosine_similarity(data)
print(f'{(similarity)}')
``` 

#### Similarity b/w two arrays

```python 
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
```

### 2 | k-means

```python 


