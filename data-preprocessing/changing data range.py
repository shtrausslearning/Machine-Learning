''' MinMax Scaler '''
# when we need to scale our data, fit/transform or fit_transform

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())
display(df_diab.head())

# Scaling columns, default (0,1)
scaler = MinMaxScaler(feature_range=(0,100))
transformed = scaler.fit_transform(df_diab)
df_data_scaled = pd.DataFrame(transformed)
display(df_data_scaled.head())
