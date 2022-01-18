class Representatives:
    def extend_df_by_cluster_path_group(self, df):
        all_cluster_paths = (
            df.groupby(["object_id"])
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

        df_cluster_path_group_extended = df.merge(
            group_assignments, on="object_id", how="left"
        )
        return df_cluster_path_group_extended

    def get_centroids(self, df):
        max_object_id=df.object_id.max()
        centroid_df = (
            df.drop(columns=["object_id"])
            .groupby(["time", "group_id", "cluster_id"])
            .mean()
            .reset_index()
            .rename(columns={"group_id": "object_id"})           
        )        
        centroid_df['object_id']=centroid_df['object_id'].apply(lambda x:x+max_object_id+1)
        
        return centroid_df

    def get_representatives(self, df, representative_type="centroids"):
        extended_df = self.extend_df_by_cluster_path_group(df)  
       
        if representative_type == "centroids":
            return self.get_centroids(extended_df)

