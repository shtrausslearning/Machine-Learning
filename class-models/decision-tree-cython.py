''' Decision Tree based Regressor '''
# more optimal MSE error calculation variant

import os
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
from sklearn import datasets
from sklearn.metrics import mean_squared_error as mse
from sklearn.tree import DecisionTreeRegressor
import itertools
%matplotlib inline
%load_ext Cython

%%cython -a 

import numpy as np
cimport numpy as np

cdef class DecisionTree:
    
    cdef public int max_depth # Number of split levels
    cdef public int feat_id   # 
    cdef public int min_size
    cdef public np.float64_t feat_thresh  
    cdef public np.float64_t value
    cdef public np.float64_t val_thresh
    cdef DecisionTree left
    cdef DecisionTree right
    
    def __init__(self, max_depth=3, min_size=4,val_thresh = 1e-5):
        
        self.max_depth = max_depth
        self.min_size = min_size
        self.val_thresh = val_thresh 
        self.left = None
        self.right = None
        self.value = 0
        self.feat_id = -1    # best feature id
        self.feat_thresh = 0  # best feature value
    
    def fit(self, np.ndarray[np.float64_t,ndim=2] X, 
                  np.ndarray[np.float64_t,ndim=1] y):

        cdef np.float64_t mean1 = 0.0
        cdef np.float64_t mean2 = 0.0
        cdef long nL = X.shape[0]
        cdef long nR = 0
        cdef np.float64_t delL = 0.0
        cdef np.float64_t delR = 0.0
        cdef np.float64_t smL = 0.0
        cdef np.float64_t smR = 0.0
        cdef long idx = 0
        
        cdef np.float64_t tot_err = 0.0
        cdef np.float64_t err_prevL = 0.0
        cdef np.float64_t err_prevR = 0.0
        cdef long thres = 0
        cdef np.float64_t error = 0.0
        
        cdef np.ndarray[long,ndim=1] idxs
        cdef np.float64_t x = 0.0
            
        # Check if max depth
        if self.max_depth <= 1:
            return
            
        # Initial Condition 
        self.value = y.mean()
        err_base = ((y - self.value)**2).sum() # start error
        error = err_base
        lhs_val, rhs_val = 0,0
        
        # Cycle through features 
        for feat in range(X.shape[1]):
            
            err_prevL, err_prevR = err_base, 0 
            idxs = np.argsort(X[:, feat])  # ascsort indicies
            mean1, mean2 = y.mean(), 0
            smL, smR = y.sum(), 0
            nL,nR = X.shape[0],0
            thres = 1
            
            # N-1
            while(thres < X.shape[0]-1):
                
                nL-=1; nR+=1
                idx = idxs[thres] 
                x = X[idx,feat]  # feature value
                
                delL = (smL - y[idx])/nL - mean1
                delR = (smR + y[idx])/nR - mean2

                smL -= y[idx]
                smR += y[idx]
                
                err_prevL += nL*(delL**2) - ( 2*delL*(smL-mean1*nL) + (y[idx]-mean1)**2 )
                err_prevR += nR*(delR**2) - ( 2*delR*(smR-mean2*nR) - (y[idx]-mean2)**2 )
                
                mean1 = smL/nL
                mean2 = smR/nR
                
                if(np.abs(x - X[idxs[thres+1],feat]) < self.val_thresh):
                    thres += 1
                    continue
                
                tot_err = err_prevL + err_prevR
                # Redefine the best feature
                if (tot_err < error):
                    if (min(nL,nR) > self.min_size):
                        self.feat_id, self.feat_thresh = feat, x
                        lhs_val, rhs_val = mean1, mean2
                        error = tot_err
                                     
                thres += 1
        
        # Nothing was split
        if(self.feat_id == -1):
            return
        
        # Instantiate Children
        self.left = DecisionTree(self.max_depth-1)
        self.right = DecisionTree(self.max_depth-1)
        self.left.value = lhs_val
        self.right.value = rhs_val
        # New Indicies
        idxs_l = X[:,self.feat_id] > self.feat_thresh
        idxs_r = X[:,self.feat_id] <= self.feat_thresh
        # Train Children
        self.left.fit(X[idxs_l,:], y[idxs_l]) 
        self.right.fit(X[idxs_r,:], y[idxs_r])
        
    def __predict(self, np.ndarray[np.float64_t, ndim=1] x):
        
        if(self.feat_id == -1):
            return self.value
        
        if(x[self.feat_id] > self.feat_thresh):
            return self.left.__predict(x)
        else:
            return self.right.__predict(x)
        
    def predict(self, np.ndarray[np.float64_t, ndim=2] X):
            
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = self.__predict(X[i])
            
        return y
      
# Evaluation Function

from sklearn.model_selection import train_test_split as tts
from sklearn import datasets

def eval_tts(ldf,feature='target',subset='train',ratio=0.3,model=None):
    
    # Split feature/target variable
    y = ldf[feature].copy()
    X = ldf.copy()
    del X[feature]     # remove target variable
    
    X = X.values; y=y.values
    X_train,X_test,y_train,y_test = tts(X,y,test_size=ratio,
                                            random_state=32)
    
    ym_train = model.fit(X_train,y_train) 
    if(subset is 'train'):
        y_model = model.predict(X_train)
        y_truth = y_train
        x_dir = [i for i in range(0,y_train.shape[0])]
    else:
        y_model = model.predict(X_test)
        y_truth = y_test
        x_dir = [i for i in range(0,y_test.shape[0])]
        
    ''' Plot Training & Model Data '''
    model_name = type(model).__name__
    
    fig = go.Figure()
    x_tr = [i for i in range(0,y_train.shape[0])]
    fig.add_traces(go.Scatter(x=x_dir,y=y_truth,mode='lines',line={'color':'#007CFF'}))
    fig.add_traces(go.Scatter(x=x_dir,y=y_model,mode='lines',line={'color':'#2C3E50'}))
    fig.update_layout(template='plotly_white',
                      margin=dict(l=20, r=20, t=120, b=20),
                      title=f'{model_name} | {subset} | Model & Truth')
    fig.update_yaxes(range=[0,50])
    fig.update_layout(showlegend=False,height=300)
    fig.show()
    
def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df_boston = sklearn_to_df(datasets.load_boston())
y_truth = df_boston['target']

model = DecisionTree(max_depth=10)
%time y_model = eval_tts(df_boston,feature='target',ratio=0.1,model=model)

model2 = DecisionTreeRegressor(max_depth=10)
%time y_model = eval_tts(df_boston,feature='target',subset='test',ratio=0.3,model=model2)

# model = CDTRegressor(max_depth=10,min_size=10)
model = DecisionTree(max_depth=10)
%time y_model = eval_tts(df_boston,feature='target',subset='test',ratio=0.3,model=model)
