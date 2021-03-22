import pandas as pd


class HistOutlier:

    def __init__(self):
        return

    @staticmethod
    def calc_outlier_degree(rating: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Params:
            rating (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time', 'cluster_end_time', 'rating'
                               containing the outlier rating for all subsequences
            clusters (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id' containing objects,
                               timestamps, cluster belongings, features ..
                               Note: The first three columns can have custom names as long as they represent the object
                               identifier, the timestamp and the cluster identifier in the right order
        Returns:
            result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time', 'cluster_end_time', 'rating', distance
                               containing the outlier rating and distances for all subsequences
        """
        object_column = clusters.columns[0]
        time_column = clusters.columns[1]
        cluster_column = clusters.columns[2]
        result = pd.DataFrame()
        ref_sequence_clusters = rating\
            .groupby(['start_time', 'end_time', 'cluster_end_time']).max()

        for index, row in ref_sequence_clusters.iterrows():
            sequence_cluster_distance = rating[(rating['start_time'] == index[0])
                                                  & (rating['end_time'] == index[1])
                                                  & (rating['cluster_end_time'] == index[2])]
            sequence_cluster_distance = sequence_cluster_distance.assign(distance = lambda x: row['rating'] - x['rating'])
            result = result.append(sequence_cluster_distance)

        times = clusters[time_column].unique()
        times = times.tolist()
        times.sort()
        for object in clusters[object_column].unique():
            object_data = clusters[clusters[object_column] == object]
            object_data.sort_values(by=[time_column])
            object_times = object_data[time_column].tolist()
            # Iterate trough all possible times in the dataframe
            for time_point in times:
                #check if the object has data for the time point
                if time_point in object_times:
                    # check if the time_point is not the last time_point
                    if times.index(time_point) + 1 < len(times):
                        # check if the followup time point and the time point is in object_times
                        followup_timepoint = times[times.index(time_point)+1]
                        if (followup_timepoint in object_times) and time_point in object_times:
                            #check if the object is an outlier at both timepoints:
                            current_outlier = clusters[
                                (clusters[time_column] == time_point) & (clusters[object_column] == object)][
                                cluster_column].tolist()[0]
                            followup_outlier = clusters[
                                (clusters[time_column] == followup_timepoint) & (clusters[object_column] == object)][
                                cluster_column].tolist()[0]

                            if (followup_outlier == -1) and (current_outlier == -1):
                                result = result.append(pd.Series([object, time_point, followup_timepoint, -1, -1, -1],
                                                                 [object_column, 'start_time', 'end_time',
                                                                  'cluster_end_time', 'rating',
                                                                  'distance']), ignore_index=True)
        return result

















