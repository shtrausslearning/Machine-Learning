import numpy as np
import pandas as pd
import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Single Plotly Timeseries Line Plot
# Input x axis -> date_time # lst-> column plots # sec_id -> secondary axis identifier T/F list

''' SINGLE PLOTLY PLOT '''
# single axis or multiple axis 

def plot_line(ldf,lst,title='',sec_id=None,size=[350,1000]):
    
    # sec_id - list of [False,False,True] values of when to activate supblots; same length as lst
    
    if(sec_id is not None):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    else:
        fig = go.Figure()
        
    if(len(lst) is not 1):
        ii=-1
        for i in lst:
            ii+=1
            if(sec_id is not None):
                fig.add_trace(go.Scatter(x=ldf.index, y=ldf[lst[ii]],mode='lines',name=lst[ii],line=dict(width=2.0)),secondary_y=sec_id[ii])
            else:
                fig.add_trace(go.Scatter(x=ldf.index, y=ldf[lst[ii]],mode='lines',name=lst[ii],line=dict(width=2.0)))
    else:
        fig.add_trace(go.Scatter(x=ldf.index, y=ldf[lst[0]],mode='lines',name=lst[0],line=dict(width=2.0)))

    fig.update_layout(height=size[0],width=size[1],template='plotly_white',title=title,
                          margin=dict(l=50,r=80,t=50,b=40));fig.show()

# data sample
nperiods = 200
np.random.seed(123)
df = pd.DataFrame(np.random.randint(-10, 12, size=(nperiods, 4)),columns=list('ABCD'))
datelist = pd.date_range(datetime.datetime(2020, 1, 1).strftime('%Y-%m-%d'),periods=nperiods).tolist()
df['dates'] = datelist 
df = df.set_index(['dates'])
df.index = pd.to_datetime(df.index)
df.iloc[0] = 0
df = df.cumsum().reset_index()

# Example
plot_line(df,['A','B'],sec_id=[False,False])

# Example: Multiple axis (for different scales)
df['B'] = df['B'] * 100
plot_line(df,['A','B'],sec_id=[False,True])
