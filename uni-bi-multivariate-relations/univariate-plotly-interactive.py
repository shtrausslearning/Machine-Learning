''' Univariate & Interative Statistics '''
# Univariate data plots using Plotly module, 
# Useful for interactive visualisation 
# Column based statistics of violin,box & histogram plots

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

''' Plot Univariate Statistics using Plotly '''
# Input Feature Matrix Dataframe

def px_stats(df, n_cols=4, to_plot='box',height=800):
    
    ldf = df.select_dtypes(include=['float64','int64'])    
#     ldf = df.select_dtypes(exclude=['float64','int64'])

    numeric_cols = ldf.columns
    n_rows = -(-len(numeric_cols) // n_cols)  # math.ceil in a fast way, without import
    row_pos, col_pos = 1, 0
    fig = make_subplots(rows=n_rows, cols=n_cols,subplot_titles=numeric_cols.to_list())
    
    for col in numeric_cols:
        if(to_plot is 'histogram'):
            trace = go.Histogram(x=ldf[col],showlegend=False,nbinsx=40)
        else:
            trace = getattr(px, to_plot)(ldf[col],x=ldf[col])["data"][0]
            
        if col_pos == n_cols: 
            row_pos += 1
        col_pos = col_pos + 1 if (col_pos < n_cols) else 1
        fig.add_trace(trace, row=row_pos, col=col_pos)

    fig.update_layout(template='plotly_white',
                      title=f'<b>UNIVARIATE FEATURE DISTRIBUTION</b>',
                      font=dict(family='sans-serif',size=12))
    
    if(to_plot is 'histogram'):
        fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    fig.update_layout(height=height);fig.show()
    
# Dataset Usage Examples
from sklearn import datasets

# convert sklearn dataset to pandas DataFrame
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_diab = sklearn_to_df(datasets.load_diabetes())
display(df_diab.head())

# Basic Univariate Data Feature Visualisation
px_stats(df_diab, to_plot='histogram') # interactive
px_stats(df_diab, to_plot='violin') # interactive
px_stats(df_diab, to_plot='box') # interactive
