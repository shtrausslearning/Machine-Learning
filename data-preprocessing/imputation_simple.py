''' Simple Imputations '''
# (1) Mean Values in column
# (2) Median Values in column
# (3) Most Frequent Values in column
# (4) Constant Fill, one value for missing data in column

from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns;sns.set(style='whitegrid')

# show na count for all columns
def show_na(X):
    fig, ax = plt.subplots(figsize = (10,5))
    nan_val = (X.isnull().sum()/len(X)*100).sort_values(ascending = False)
    cmap = sns.color_palette("plasma")
    ax.spines['top'].set_visible(True);ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True);ax.spines['left'].set_visible(True)
    sns.barplot(x=nan_val,y=nan_val.index, edgecolor='w',palette = cmap)
    plt.title('Missing Data in DataFrame (%)');plt.show()

# function to add na into columns feat
def make_na(df,feat,lst_p):

    ii=-1
    for i in feat:
        ii+=1;df[i] = df[i].apply(lambda x: np.nan if np.random.rand() < lst_p[ii] else x)
    return df

# function to load example dataset
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

# load dataset 
df_diab = sklearn_to_df(datasets.load_diabetes())
display(df_diab.head())

# make na data for bmi and age columns
df_diab_na = make_na(df_diab,['bmi','age'],[2/10,2/4])
show_na(df_diab_na)

# by default imputes w/ column mean values
imputed_mean = SimpleImputer()
imp = imputed_mean.fit_transform(df_diab_na)

# impute with column median
imputed_median = SimpleImputer(strategy='median')
imp = imputed_median.fit_transform(df_diab_na)

# impute using most frequent value in each column
imputed_freq = SimpleImputer(strategy='most_frequent')
imp = imputed_freq.fit_transform(df_diab_na)

# impute using most frequent value in each column
imputed_const = SimpleImputer(strategy='constant',fill_value=1)
imp = imputed_const.fit_transform(df_diab_na)
