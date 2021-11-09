''' Function plots & returns highest weighted features '''
# for SVC linear covariance function model in SQ sequence format

def fi_svc(classifier, feature_names, top_features=5,verbose=False):
    
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    # plt.title("Feature Importances (Support Vector Machine) - Ciprofloxacin Resistance", y=1.08)
    colors = ['crimson' if c < 0 else 'cornflowerblue' for c in coef[top_coefficients]]
    feature_names = np.array(feature_names)
    lser = pd.Series(data=coef[top_coefficients],index=feature_names[top_coefficients])
    fig = px.bar(lser,orientation='h')
    fig.update_traces(width=0.5)
    fig.update_layout(height=350,template='plotly_white',showlegend=False,
                        title=f"<b>FEATURE IMPORTANCE</b> | SVC")
    fig.show()
    
    # if we print the unitigs, we can then look at what genes they relate to
    top_negative_coefficients = np.argsort(coef)[:5]
    neg_predictors = np.asarray(feature_names)[top_negative_coefficients]
    top_positive_coefficients = np.argsort(coef)[-5:]
    pos_predictors = np.asarray(feature_names)[top_positive_coefficients]
    if(verbose):
        print("Top negative predictors: ",neg_predictors)
        print("Top positive predictors: ",pos_predictors)
    
    # Store the most important features
    top_negSeq = []; top_posSeq = []
    for i in range(0,top_features):
        top_negSeq.append(Seq(neg_predictors[i]))
        top_posSeq.append(Seq(pos_predictors[i]))
        
    return top_negSeq, top_posSeq 
