import plotly.graph_objects as go
import plotly
from itertools import cycle
import pandas as pd

class Plotter2d:
    def __init__(self, df, x_col="time", y_col="feature1"):
        self.fig_3d = go.Figure()
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
      
        self.fig_3d.update_layout(
            scene=dict(
                xaxis_title=self.x_col, yaxis_title=self.y_col
            ),         
        )

    def add_representatives(self, df):
        df["representative"] = True
        merged_df = pd.concat([self.df, df]).reset_index(drop=True)
        self.df = merged_df

    def generate_lines(self):
        ts_group_condition = ["object_id"]
        if "representative" in self.df.columns:
            ts_group_condition.append("representative")
        ts = (
            self.df.groupby(ts_group_condition, dropna=False)
            .agg(lambda x: list(x))
            .reset_index()
        )
        for index, row in ts.iterrows():
            line_width = 3
            legendgroup = "time_series"
            legendgrouptitle_text = "Time series"
            dash = "solid"
            if "representative" in row and row["representative"] == True:
                line_width = 10
                legendgroup = "representative"
                legendgrouptitle_text = "Representative time series"
                dash = "dash"

            self.fig_3d.add_trace(
                go.Scatter(
                    x=row[self.x_col],
                    y=row[self.y_col],
                  
                    mode="lines",
                    line=dict(width=line_width, dash=dash),
                    name=f"time series {row['object_id']}",
                    legendgroup=legendgroup,
                    legendgrouptitle_text=legendgrouptitle_text,
                )
            )

    def generate_markers(self):
        colors = cycle(plotly.colors.qualitative.Plotly)
        cluster_group_condition = ["cluster_id"]

        clusters = (
            self.df.groupby(cluster_group_condition, dropna=False)
            .agg(lambda x: list(x))
            .reset_index()
        )
        for index, row in clusters.iterrows():
            marker_size = 10
            marker_color = next(colors)
            self.fig_3d.add_trace(
                go.Scatter(
                    x=row[self.x_col],
                    y=row[self.y_col],                    
                    mode="markers",
                    marker=dict(size=marker_size, color=marker_color),
                    name=f"cluster {row['cluster_id']}",
                    legendgroup="clusters",
                    legendgrouptitle_text="Clusters",
                )
            )

    def generate_fig(self):
        self.generate_lines()
        self.generate_markers()
