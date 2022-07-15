
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
