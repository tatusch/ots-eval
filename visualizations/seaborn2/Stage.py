import seaborn as sns

class Stage:
    def __init__(self,
            df_cluster_groups, 
            df_mapping=dict(
            time_col="time",
                object_id_col="object_id",
                f1_col="feature1",
                f2_col="feature2",
                group_col="group_id",
            ),
            plot_settings=dict(col_wrap=3, bbox_to_anchor=(-1.2, -0.4))):
        
        self.df_cluster_groups=df_cluster_groups
        self.col_wrap=plot_settings['col_wrap']
        self.time_col=df_mapping['time_col']
    
    def getStage(self):
        sns.set(style="darkgrid")
        g = sns.FacetGrid(
            col_wrap=self.col_wrap,
            data=self.df_cluster_groups,
            col=self.time_col,
            palette="Set1",
            hue="group_id",
        )
        return g
