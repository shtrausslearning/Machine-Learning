''' Robust Scaler '''
# When we know that we have outliers in our data, we can choose to
# remove their effect during the scaling using RobustScaler

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())
display(df_diab.head())

# Using the quantile_range parameter, by default uses IQR = (25,75) for scale mod
scaler = RobustScaler(quantile_range=(25,75))
transformed = scaler.fit_transform(df_diab)
df_data_scaled = pd.DataFrame(transformed)
display(df_data_scaled.head())
