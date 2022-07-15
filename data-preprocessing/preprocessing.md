
- <code>label</code> encoding

```
import pandas as pd

non_categorical_series = pd.Series(['male', 'female', 'male', 'female'])  # Create series with male and female values
categorical_series = non_categorical_series.astype('category') # Convert the text series to a categorical series

print(categorical_series.cat.codes) # Print the numeric codes for each value
```

```
0    1
1    0
2    1
3    0
dtype: int8
```

- <code>one-hot encoding</code> with pandas

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

