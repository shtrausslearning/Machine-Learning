
### Data Preprocessing

#### 1 | Categorical Data

- <code>label</code> encoding

```
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

- <code>changing feature range.py</code> - Изменения диапазона признака от A до B (ie. нормализация), <code>MinMaxScalar</code>
- <code>feature_reduction.py</code> - Уменьшения количество признаков истользуя методы неконтролируемого обучения
- <code>imputation_model.py</code> - Заполнение пропущенных значений используя модели машинного обучения
- <code>imputation_simple.py</code> - Заполнение пропущенных значений используя <code>SimpleImputer</code>
- <code>normalising instances.py</code> - 
- <code>scaling with outliers.py</code> - 
- <code>standardising data.py</code> - 
- <code>standardscaling.py</code> - Масштабирование признаков <code>StandardScaler</code>, <code>MinMaxScaler</code>
