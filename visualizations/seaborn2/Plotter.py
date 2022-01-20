from Stage import Stage
from Subplot import Subplot
from Legend import Legend
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
        self.df_mapping=df_mapping
        self.plot_settings=plot_settings

    def _extend_df_by_cluster_path_group(self):
        all_cluster_paths = (
            self.df.groupby(["object_id"])
            .cluster_id.apply(tuple)
            .reset_index(name="cluster_path")
        )

        cluster_path_groups = (
            all_cluster_paths.drop(columns=["object_id"])
            .drop_duplicates("cluster_path")
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_id"})
        )

        group_assignments = cluster_path_groups.merge(
            all_cluster_paths, on="cluster_path", how="left"
        ).drop(columns=["cluster_path"])

        df_cluster_path_group_extended = self.df.merge(
            group_assignments, on="object_id", how="left"
        )
        return df_cluster_path_group_extended


    def generate_fig(self):
        df_cluster_groups = self._extend_df_by_cluster_path_group()
        stage=Stage(df_cluster_groups,self.df_mapping,self.plot_settings)
        subplot=Subplot(self.df_mapping)
        legend=Legend(df_cluster_groups,self.plot_settings)
        
        g=stage.getStage()        
        rp_contained = True if "representative" in self.df.columns else False        
        g=subplot.addSubplots(g,rp_contained)
        legend.addLegend(plt,rp_contained=rp_contained)
        return g

    def add_representatives(self, df):
        df["representative"] = True
        merged_df = pd.concat([self.df, df]).reset_index(drop=True)
        self.df = merged_df

