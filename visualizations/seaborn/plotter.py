from tokenize import group
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#rng = np.random.RandomState(0)
#x = np.linspace(0, 10, 500)
#y = np.cumsum(rng.randn(500, 6), 0)

class Plotter:
    def __init__(self,df,df_mapping=dict(time_col='time',object_id_col='object_id', f1_col='feature1', f2_col='feature2', group_col='group_id')):
        self.df=df
        self.time_col=df_mapping['time_col']
        self.f1_col=df_mapping['f1_col']
        self.f2_col=df_mapping['f2_col']
        self.group_col=df_mapping['group_col']
        self.object_id_col=df_mapping['object_id_col']

    def extend_df_by_cluster_path_group(self):
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
        def f(x,y,z,r='',**kwargs):
            ax = sns.scatterplot(x=x,y=y, **kwargs, s=0)

            for i in range(len(x)):
                ax.annotate(z.values[i], xy=(x.values[i], y.values[i]), fontsize=15,
                            xytext=(0, 0), textcoords="offset points",
                            bbox=dict(boxstyle='round', alpha=0.3,color=kwargs['color']),
                            va='center', ha='center', weight='bold', alpha=1)
            return
        df_cluster_groups=self.extend_df_by_cluster_path_group()
        
        g=sns.FacetGrid(data=df_cluster_groups,col=self.time_col, palette='Set1', hue='group_id')            
        if 'representative' in self.df.columns:
            g.map(f,self.f1_col,self.f2_col, self.object_id_col,'representative')
        else:
            g.map(f,self.f1_col,self.f2_col, self.object_id_col)
        g.add_legend()
        return g
    
    def add_representatives(self, df):
        df['representative'] = True
        merged_df = pd.concat([self.df,df]).reset_index(drop=True)
        self.df=merged_df
