import plotly.express as px


class Plotter:
    def __init__(self):
        #self.df=df
        self.fig_3d=None
        self.fig_2d=None

    def generate_fig_2d(self,df):
        fig = px.line(df, x="time", y="feature1", color="object_id")
        fig.add_traces(
            list(
                px.scatter(
                    df, x="time", y="feature1", color="cluster_id"
                ).select_traces()
            )
        )
        fig.update_traces(marker={"size": 15})
        fig.update_layout(
            legend=dict(
                yanchor="top",
                xanchor="right",
            )
        )
        self.fig_2d=fig
        

    def generate_fig_3d(self,df):
        fig = px.line_3d(df, x="time", y="feature1", z="feature2", color="object_id")
        fig.add_traces(
            list(
                px.scatter_3d(
                    df, x="time", y="feature1", z="feature2", color="cluster_id"
                ).select_traces()
            )
        )
        fig.update_xaxes(type="category")
        fig.update_layout(
            legend=dict(
                yanchor="top",
                xanchor="right",
            )
        )
        self.fig_3d=fig
        
    def add_representatives_3d(self,df):
        self.fig_3d.add_traces(
           list(
                px.scatter_3d(
                    df, x="time", y="feature1", z="feature2", color="cluster_id"
                ).select_traces()
        )
        )
        self.fig_3d.add_traces(
            list(px.line_3d(df, x='time', y='feature1', z='feature2', color='object_id',width=40,labels={'object_id':'representative_id'}).select_traces())
            
            )
