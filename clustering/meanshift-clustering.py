''' Meanshift Clustering '''
# together w/ dbscan, don't require specification of number of clusters
# automatically choses how many are needed by looking at data & group
# groups are still assumed to be of blob nature

from sklearn.cluster import MeanShift

data = np.array([
  [5.7, 5.5, 1.4, 0.15],
  [4.6, 3.7 , 7.4, 0.16],
  [4.7, 3.9, 6.3, 0.66],
  [4.9, 3.4, 2.5, 0.72],
  [5.2 , 4.1, 3.4, 0.32]])

# Meanshift Model
model = MeanShift()
model.fit(data)

print(f'labels: {model.labels_}') # cluster assignment
print(model.cluster_centers_) # cluster centroids

# new observation
new_obs = np.array([
  [5.1, 3.2, 1.7, 1.9],
  [6.9, 3.2, 5.3, 2.2]])

# predict clusters
prediction = model.predict(new_obs)
print(f'prediction: {prediction}')
