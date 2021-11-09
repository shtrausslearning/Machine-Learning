''' Cluster based Feature Reduction '''
# Like other unsupervised learning methods, we can
# use clustering to group together data & reduce features
# using FeatureAgglomeration
from sklearn.cluster import FeatureAgglomeration

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

print(f'Input Data: {data.shape}')
model = FeatureAgglomeration(n_clusters=3)
dr_data = model.fit_transform(data)
print(f'DR shape: {dr_data.shape}')
