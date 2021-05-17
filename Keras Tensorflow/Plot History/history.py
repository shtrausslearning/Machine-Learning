# Function to plot all metrics side by side (defined above)
def plot_keras_metric2(lst_history,names):

    # Palettes
    lst_color = ['#B1D784','#2E8486','#004379','#032B52','#EAEA8A']
    metric_id = ['get_f1','acc','get_precision','get_recall']

    ii=-1
    for i in range(len(lst_history)):

        ii+=1; history = lst_history[ii]
        fig = make_subplots(rows=1, cols=4,subplot_titles=metric_id)

#       Change the metric_id to whatever you want to display, just change the subplot_titles
        jj=0
        for metric in metric_id:     

            jj+=1

            # Main Trace
            fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=history.history[metric],
                                     name=f'train_{metric}_{ii}',line=dict(color=lst_color[3]),mode='lines'),
                          row=1,col=jj)
            fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=history.history['val_'+metric],
                                     name=f'valid_{metric}_{ii}',line=dict(color=lst_color[1]),mode='lines'),
                          row=1,col=jj)

            # difference between training/validation metrics
            diff = abs(np.array(history.history[metric]) - np.array(history.history['val_'+metric]))
            fig.add_trace(go.Bar(x=[i for i in range(1,n_epochs+1)],y=diff,name='metric diff.',
                                 text=diff.round(3),marker_color=lst_color[2],opacity=0.25,showlegend=False)
                          ,row=1,col=jj)

            # Rest are for reference only
            for j in range(len(names)):
                if(j != ii):
                    lhistory = lst_history[j]
                    fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=lhistory.history[metric],
                                             name=f'train_{metric}_{j}',opacity=0.1,showlegend=False,
                                             line=dict(color=lst_color[3]),mode='lines'),row=1, col=jj)
                    fig.add_trace(go.Scatter(x=[i for i in range(1,n_epochs+1)],y=lhistory.history['val_'+metric],
                                             name=f'val_{metric}_{j}',opacity=0.1,showlegend=False,
                                             line=dict(color=lst_color[1]),mode='lines'),row=1, col=jj)

        fig.update_yaxes(tickvals=[i for i in np.arange(0.0,1.0,0.1)])
        fig.update_layout(yaxis=dict(range=[0,1]))
        fig.update_layout(yaxis_range=[0,1])
        fig.update_layout(template='plotly_white',font=dict(family='sans-serif',size=14))
        fig['layout']['yaxis1'].update(title='', range=[0,1], autorange=False, tick0=0, dtick=5)
        fig['layout']['yaxis2'].update(title='', range=[0,1], autorange=False, tick0=0, dtick=5)
        fig['layout']['yaxis3'].update(title='', range=[0,1], autorange=False, tick0=0, dtick=5)
        fig['layout']['yaxis4'].update(title='', range=[0,1], autorange=False, tick0=0, dtick=5)
        if(len(names)>1):
            title=f"{' | '.join(str(x) for x in names[ii])}"
            if(type(names[0]) is not list):
                title= names[ii]
        else:
            title= f"{''.join(str(x) for x in names[ii])}"
        fig.update_layout(title=title,height=300,showlegend=False)

        fig.add_hline(y=1.0,line_width=5)
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        fig.show()
