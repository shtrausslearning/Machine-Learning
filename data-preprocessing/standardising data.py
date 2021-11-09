from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())
display(df_diab.head())

# Standardizing each column of dataframe
np_data_stand = scale(df_diab)
df_data_stand = pd.DataFrame(np_data_stand)
display(df_data_stand.head())

# Column means (rounded to nearest thousandth)
col_mean = np_data_stand.mean(axis=0).round(decimals=3)
print(f'mean of each column: {col_mean}')

# Column standard deviations
col_stds = np_data_stand.std(axis=0)
print(f'column standard deviation: {col_stds}')
