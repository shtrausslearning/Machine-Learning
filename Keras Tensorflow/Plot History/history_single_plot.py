''' FUNCTIONS FOR PLOTTING KERAS HISTORY RESULTS '''
# Classification based visualisation of multiple metrics defined in metric_id
# Focus on comparing training & evaluation for one case only

# Function to plot all metrics side by side (defined above)
def plot_keras_metric(history):

    # Palettes
    lst_color = ['#B1D784','#2E8486','#004379','#032B52','#EAEA8A']
    metric_id = ['loss','get_f1','acc','get_precision','get_recall']

    fig = make_subplots(rows=1, cols=len(metric_id),subplot_titles=metric_id)

#       Change the metric_id to whatever you want to display, just change the subplot_titles
    jj=0;
    for metric in metric_id:     

        jj+=1

        # Main Trace
        fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=history.history[metric],
                                 name=f'train_{metric}',line=dict(color=lst_color[3]),mode='lines'),
                      row=1,col=jj)
        fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=history.history['val_'+metric],
                                 name=f'valid_{metric}',line=dict(color=lst_color[1]),mode='lines'),
                      row=1,col=jj)

        # difference between training/validation metrics
        if(metric is not 'loss'):
            diff = abs(np.array(history.history[metric]) - np.array(history.history['val_'+metric]))
            fig.add_trace(go.Bar(x=[i for i in range(1,n_epochs+1)],y=diff,name='metric diff.',
                                 text=diff.round(3),marker_color=lst_color[2],opacity=0.25,showlegend=False)
                          ,row=1,col=jj)

    fig.update_layout(yaxis=dict(range=[0,1]))
    fig.update_layout(yaxis_range=[0,1])
    fig.update_layout(template='plotly_white')
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=30),height=300,showlegend=False)
    fig['layout']['yaxis'].update(title='', range=[0,5], autorange=True,type='log')
    fig['layout']['yaxis2'].update(title='', range=[0, 1.1], autorange=False)
    fig['layout']['yaxis3'].update(title='', range=[0, 1.1], autorange=False)
    fig['layout']['yaxis4'].update(title='', range=[0, 1.1], autorange=False)
    fig['layout']['yaxis5'].update(title='', range=[0,1.1],autorange=False)
    fig.update_layout(hovermode="x") # add comparison at epoch
    fig.show()
