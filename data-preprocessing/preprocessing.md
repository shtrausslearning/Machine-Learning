
### Data Preprocessing

#### 1 | Categorical Data

- <code>label</code> encoding of a **list**

```python
import pandas as pd

non_categorical_series = pd.Series(['male', 'female', 'male', 'female']) 
categorical_series = non_categorical_series.astype('category') 
print(categorical_series.cat.codes) 
```

```
0    1
1    0
2    1
3    0
dtype: int8
```

- <code>one-hot encoding</code> with pandas using <code>get_dummies</code>

```python
non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])
print(pd.get_dummies(non_categorical_series))
```

```
   female  male
0       0     1
1       1     0
2       0     1
3       1     0
```

- <code>minmax scaling</code>

```python
# Scaling columns, default (0,1)
scaler = MinMaxScaler(feature_range=(0,100))
transformed = scaler.fit_transform(df_diab)
df_data_scaled = pd.DataFrame(transformed)
print(df_data_scaled.head().to_markdown())
```

```
|    |        0 |   1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |
|---:|---------:|----:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
|  0 | 66.6667  | 100 | 58.2645 | 54.9296 | 29.4118 | 25.6972 | 20.7792 | 28.2087 | 56.2217 | 43.9394 | 39.2523 |
|  1 | 48.3333  |   0 | 14.876  | 35.2113 | 42.1569 | 30.6773 | 62.3377 | 14.1044 | 22.2443 | 16.6667 | 15.5763 |
|  2 | 88.3333  | 100 | 51.6529 | 43.662  | 28.9216 | 25.8964 | 24.6753 | 28.2087 | 49.6584 | 40.9091 | 36.1371 |
|  3 |  8.33333 |   0 | 30.1653 | 30.9859 | 49.5098 | 44.7211 | 23.3766 | 42.3131 | 57.2936 | 46.9697 | 56.3863 |
|  4 | 51.6667  |   0 | 20.6612 | 54.9296 | 46.5686 | 41.7331 | 38.961  | 28.2087 | 36.2369 | 33.3333 | 34.2679 |
```
