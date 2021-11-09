''' Cluster Evaluations metrics '''
# Adjusted Rand Score & Adjusted Mutual Info Score 
# cluster evaluation metric using labels

# general rules of thumb
# ARC is used when the true clusters are large and approx equal sized, 
# AMI is used when the true clusters are unbalanced in size and there exist small clusters.

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score

truth = np.array([0, 0, 0, 1, 1, 1])
pred = np.array([0, 0, 1, 1, 2, 2])
best = np.array([0, 0, 0, 1, 1, 1])
permuted_best = np.array([1, 1, 1, 0, 0, 0])
similar_pred = np.array([1, 1, 1, 3, 3, 3])

# ARS, symmetric as well
ar = adjusted_rand_score(truth, pred)
ami = adjusted_mutual_info_score(truth, pred)
print(f'adjusted_rand_score: {ar}')
print(f'adjusted_mutual_info_score: {ami}')

# Perfect labeling
ar = adjusted_rand_score(truth, best)
ami = adjusted_mutual_info_score(truth, best)
print(f'max adjusted_rand_score: {ar}')
print(f'max adjusted_mutual_info_score: {ami}')

# permutations in the labeling
ar = adjusted_rand_score(truth, permuted_best)
ami = adjusted_mutual_info_score(truth, permuted_best)
print(f'permuted max adjusted_rand_score: {ar}')
print(f'permuted adjusted_mutual_info_score: {ami}')

# similarly labeled 
ar = adjusted_rand_score(truth, similar_pred)
ami = adjusted_mutual_info_score(truth, similar_pred)
print(f'similarly labeled adjusted_rand_score: {ar}')
print(f'similarly labeled adjusted_mutual_info_score: {ami}')
