
''' 1. MODEL ENSEMBLE IMPUTATION '''
# kNN Regressor + CatBoost Regressor Ensemble Imputation 

from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor

# function that imputes a dataframe 
def impute_model(df,cols=None):

    # separate dataframe into numerical/categorical
    ldf = df.select_dtypes(include=[np.number])           # select numerical columns in df
    ldf_putaside = df.select_dtypes(exclude=[np.number])  # select categorical columns in df
    # define columns w/ and w/o missing data
    cols_nan = ldf.columns[ldf.isna().any()].tolist()         # list of features w/ missing data 
    cols_no_nan = ldf.columns.difference(cols_nan).values     # get all colun data w/o missing data
    
    if(cols is not None):
        cols_nan = cols
        df1 = ldf[cols_nan].describe()
    
    fill_id = -1
    for col in cols_nan:    
        fill_id+=1
        imp_test = ldf[ldf[col].isna()]   # indicies which have missing data will become our test set
        imp_train = ldf.dropna()          # all indicies which which have no missing data 
        model0 = CatBoostRegressor(verbose=False)               # Catboost Regressor Supervised Approach
        model1 = KNeighborsRegressor(n_neighbors=15)            # KNR Unsupervised Approach
        knr = model0.fit(imp_train[cols_no_nan], imp_train[col])
        xgb = model1.fit(imp_train[cols_no_nan], imp_train[col])
        knrP = knr.predict(imp_test[cols_no_nan])
        xgbP = xgb.predict(imp_test[cols_no_nan])
        pred = (knrP + xgbP)*0.5 # Simple Model Ensemble
        ldf.loc[df[col].isna(), col] = pred              
        ldf.loc[df[col].isna(),'fill_id'] = fill_id   # Add imputation 
        
    df2 = ldf[cols_nan].describe()
        
    return pd.concat([ldf,ldf_putaside],axis=1)

''' 2. EXAMPLE '''

from sklearn import datasets
import pandas as pd
import numpy as np

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

''' Load DataFrame w/o missing data '''
df_boston = sklearn_to_df(datasets.load_boston())
display(df_boston.head())
#display(df_boston.head())
# df_cali = sklearn_to_df(datasets.fetch_california_housing())
#display(df_cali.head())
# df_diab = sklearn_to_df(datasets.load_diabetes())
# print(df_diab.isna().sum()) # check missing data in columns

 ''' Remove random percentage of data in two columns '''
p = 2/10
df_boston["NOX"] = df_boston["NOX"].apply(lambda x: np.nan if np.random.rand() < p else x)
p = 2/4
df_boston["AGE"] = df_boston["AGE"].apply(lambda x: np.nan if np.random.rand() < p else x)
display(df_boston.head())

# Show NaN in DataFrame (visual plot)
import matplotlib.pyplot as plt
import seaborn as sns

''' Show NaN count '''
def show_na(X):
    fig, ax = plt.subplots(figsize = (10,5))
    nan_val = (X.isnull().sum()/len(X)*100).sort_values(ascending = False)
    cmap = sns.color_palette("plasma")
    ax.spines['top'].set_visible(True);ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True);ax.spines['left'].set_visible(True)
    sns.barplot(x=nan_val,y=nan_val.index, edgecolor='w',palette = cmap)
    plt.title('Missing Data in DataFrame (Percentage)');plt.show()
    
show_na(df_boston)

''' Impute Missing Data '''
df_boston_imputed = impute_model(df_boston)
display(df_boston_imputed)
