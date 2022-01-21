class DataPreprocessor:
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
    ):
        self.df = df
        self.object_id_col = df_mapping["object_id_col"]

    def extend_df_by_cluster_path_group(self):
        all_cluster_paths = (
            self.df.groupby([self.object_id_col])
            .cluster_id.apply(tuple)
            .reset_index(name="cluster_path")
        )

        cluster_path_groups = (
            all_cluster_paths.drop(columns=[self.object_id_col])
            .drop_duplicates("cluster_path")
            .reset_index(drop=True)
            .reset_index()
            .rename(columns={"index": "group_id"})
        )

        group_assignments = cluster_path_groups.merge(
            all_cluster_paths, on="cluster_path", how="left"
        ).drop(columns=["cluster_path"])

        df_cluster_path_group_extended = self.df.merge(
            group_assignments, on=self.object_id_col, how="left"
        )
        return df_cluster_path_group_extended
