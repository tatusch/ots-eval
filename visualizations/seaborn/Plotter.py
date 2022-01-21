from visualizations.seaborn.Stage import Stage
from visualizations.seaborn.Subplot import Subplot
from visualizations.seaborn.Legend import Legend
from visualizations.seaborn.DataPreprocessor import DataPreprocessor

import matplotlib.pyplot as plt
import pandas as pd


class Plotter:
    def __init__(
        self,
        df,
        df_mapping=dict(
            time_col="time",
            object_id_col="object_id",
            f1_col="feature1",
            f2_col="feature2",
            group_col="group_id",
        ),
        plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4)),
    ):
        self.df = df
        self.df_mapping = df_mapping
        self.plot_settings = plot_settings

    def generate_fig(self):
        preprocessor = DataPreprocessor(self.df, self.df_mapping)
        df_cluster_groups = preprocessor.extend_df_by_cluster_path_group()
        stage = Stage(df_cluster_groups, self.df_mapping, self.plot_settings)
        subplot = Subplot(self.df_mapping)
        legend = Legend(df_cluster_groups, self.plot_settings)

        g = stage.getStage()
        rp_contained = True if "representative" in self.df.columns else False
        g = subplot.addSubplots(g, rp_contained)
        legend.addLegend(plt, rp_contained=rp_contained)
        return g

    def add_representatives(self, df):
        df["representative"] = True
        merged_df = pd.concat([self.df, df]).reset_index(drop=True)
        self.df = merged_df
