import plotly.graph_objects as go
import pandas as pd
import numpy as np

#class Plotter:
#    def __init__(self, df):
#        df.groupby(['object_id']).apply(tuple)
#    def generate_fig_3d(self):

def get_data():
    # test_data for 6 time series with 3 timestamps, 2 clusters per timestamp and 2 features (features are optional as those are not considered in CLOSE)
    test_data = [[1, 1, 1, 1 / 3, 1 / 6], [2, 1, 1, 2 / 3, 1 / 6], [3, 1, 1, 1 / 3, 2 / 6], [4, 1, 2, 2 / 3, 4 / 6], [5, 1, 2, 3 / 3, 4 / 6], [6, 1, 2, 2 / 3, 5 / 6],
                 [1, 2, 3, 2 / 3, 1 / 6], [2, 2, 3, 3 / 3, 1 / 6], [3, 2, 3, 2 / 3, 2 / 6], [4, 2, 4, 2 / 3, 5 / 6], [5, 2, 4, 3 / 3, 5 / 6], [6, 2, 4, 2 / 3, 6 / 6],
                 [1, 3, 5, 2 / 3, 1 / 6], [2, 3, 5, 2 / 3, 2 / 6], [3, 3, 5, 1 / 3, 1 / 6], [4, 3, 6, 2 / 3, 5 / 6], [5, 3, 6, 3 / 3, 4 / 6], [6, 3, 6, 1 / 3, 6 / 6]]

    data = pd.DataFrame(test_data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])
    return data

df=get_data()
print(df)
ts=df.groupby(['object_id']).agg(lambda x: list(x)).reset_index()
clusters=df.groupby(['cluster_id']).agg(lambda x: list(x)).reset_index()

print(ts)

fig=go.Figure()

for index,row in ts.iterrows():    
    fig.add_trace(go.Scatter3d(x=row['time'],y=row['feature1'],z=row['feature2'], mode='lines', name=f"time series {row['object_id']}",))
   

for index,row in clusters.iterrows():
    fig.add_trace(go.Scatter3d(x=row['time'],y=row['feature1'],z=row['feature2'], mode='markers', name=f"cluster {row['cluster_id']}",))
fig.show()
