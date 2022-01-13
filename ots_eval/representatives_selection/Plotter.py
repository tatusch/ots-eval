import plotly.express as px


class Plotter:
    def __init__(self, df):
        self.df = df

    def plot_2d(self):
        fig = px.line(self.df, x="time", y="feature1", color="object_id")
        fig.add_traces(
            list(
                px.scatter(
                    self.df, x="time", y="feature1", color="cluster_id"
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
        return fig

    def plot_3d(self):
        fig = px.line_3d(self.df, x="time", y="feature1", z="feature2", color="object_id")
        fig.add_traces(
            list(
                px.scatter_3d(
                    self.df, x="time", y="feature1", z="feature2", color="cluster_id"
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
        return fig