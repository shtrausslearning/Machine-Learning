''' Tree Based Feature Importance '''
# requires evaluation class input w/ at least one of RF, CatBoost & XGB moedls
# models stored in .store_models are required from eval class
# notebook @https://www.kaggle.com/shtrausslearning/identifying-antibiotic-resistant-bacteria

class fi:
    
    def __init__(self,data=None, # evaluation class
                      sort_by='RF', # show most important features
                      max_features=10 # limit unitigs to 
                ):
        
        if(data is None):
            print('Enter Evaluation class w/ CAT,RF,XGB')
        else:
            evals = data
            # check which models are present
            lst_models = list(evals.store_models.keys())
            temp = []
            for i in lst_models:
                if('CAT' in i):
                    temp.append('CAT')
                if('RF' in i):
                    temp.append('RF')
                if('XGB' in i):
                    temp.append('XGB')
                    
            # input contains gscv data
            if('GS' in lst_models[0]):
                self.gs_id = True
            else:
                self.gs_id = False
                
        self.lst_tree_models = list(set(temp))
        
        self.evals = data  # evaluation class
        self.lst_Seqs = []  # list of important unitigs
        self.max_features = max_features # show top n features
        self.sort_by = sort_by # sort by particualr model fi, other mods show this index only
        self.abr_feat = False # activate if unitig names are too big for figure

    # Compile all Tree based feature importance results
    def get(self):

        # USL scaling
        min_max_scaler = preprocessing.MinMaxScaler()
        
        # Recall Model & Get Feature Importance from data class
        # unless gridsearched, all kfolds are the same model
        
        ii=-1
        # if randomforest models are present
        if('RF' in self.lst_tree_models):
            
            if(self.gs_id):
                
                # fold names
                tlst_models = [f'GS_RF_{i}' for i in range(0,self.evals.nfold)]
                
                # stack all fold results
                for kfold_id in tlst_models: 
                    ii+=1
                    rf_model = self.evals.store_models[kfold_id]
                    imp_rf = rf_model.feature_importances_
                    rf_sc = min_max_scaler.fit_transform(imp_rf[:,None])
                    ldf = pd.DataFrame(rf_sc,index=self.evals.X.columns,columns=[kfold_id])
                    if(ii is 0):
                        df = ldf.copy()
                    else:
                        df = pd.concat([df,ldf],axis=1)
                    
            else:
            
                ii+=1
                rf_model = self.evals.store_models['RF_1']
                imp_rf = rf_model.feature_importances_
                rf_sc = min_max_scaler.fit_transform(imp_rf[:,None])
                ldf = pd.DataFrame(rf_sc,index=self.evals.X.columns,columns=['RF'])
                if(ii is 0):
                    df = ldf.copy()
                else:
                    df = pd.concat([df,ldf],axis=1)
                
        # if catboost models are present
        if('CAT' in self.lst_tree_models):
            
            if(self.gs_id):
                
                # fold names
                tlst_models = [f'GS_CAT_{i}' for i in range(0,self.evals.nfold)]
                
                # stack all fold results
                for kfold_id in tlst_models: 
                    ii+=1
                    cb_model = self.evals.store_models[kfold_id]
                    imp_cb = cb_model.get_feature_importance()
                    cb_sc = min_max_scaler.fit_transform(imp_cb[:,None])
                    ldf = pd.DataFrame(cb_sc,index=self.evals.X.columns,columns=[kfold_id])
                    if(ii is 0):
                        df = ldf.copy()
                    else:
                        df = pd.concat([df,ldf],axis=1)
                    
            else:
                ii+=1
                cb_model = self.evals.store_models['CAT_1']
                imp_cb = cb_model.get_feature_importance()
                cb_sc = min_max_scaler.fit_transform(imp_cb[:,None])
                ldf = pd.DataFrame(cb_sc,index=self.evals.X.columns,columns=['CB'])
                if(ii is 0):
                    df = ldf.copy()
                else:
                    df = pd.concat([df,ldf],axis=1)
                
            
        if('XGB' in self.lst_tree_models):
            
            if(self.gs_id):

                # fold names
                tlst_models = [f'GS_XGB_{i}' for i in range(0,self.evals.nfold)]

                # stack all fold results
                for kfold_id in tlst_models: 
                    ii+=1
                    xg_model = self.evals.store_models[kfold_id]
                    imp_xg = xg_model.feature_importances_
                    xg_sc = min_max_scaler.fit_transform(imp_xg[:,None])
                    ldf = pd.DataFrame(xg_sc,index=self.evals.X.columns,columns=[kfold_id])
                    if(ii is 0):
                        df = ldf.copy()
                    else:
                        df = pd.concat([df,ldf],axis=1)

            else:

                ii+=1
                xg_model = self.evals.store_models['XGB_1']
                imp_xg = xg_model.feature_importances_
                xg_sc = min_max_scaler.fit_transform(imp_xg[:,None])
                ldf = pd.DataFrame(rf_sc,index=self.evals.X.columns,columns=['XGB'])
                
                if(ii is 0):
                    df = ldf.copy()
                else:
                    df = pd.concat([df,ldf],axis=1)

        # change to abbrev if names are too long to display
        if(self.abr_feat):
            self.evals.col_trans(0)
        
        # Sort by one of the available columns
        df.sort_values(by=self.sort_by,ascending=False,inplace=True)

        if(self.abr_feat):
                self.evals.col_trans(1)

        # show only most critical features in FI
        subset = df[:self.max_features]
        
#       Store the most important features
        for i in subset.index.tolist():
            self.lst_Seqs.append(Seq(i))
        
        # Plot features 
        fig = px.bar(subset,orientation='h')
        fig.update_traces(width=0.5)
        fig.update_layout(height=400,template='plotly_white',
                          title=f"<b>FEATURE IMPORTANCE</b> | Sorted by {self.sort_by}()")
        fig.show()
