import numpy as np
from scipy.spatial.distance import euclidean
from typing import Union
import pandas


class CLOSE(object):

    def __init__(self, data: pandas.DataFrame, measure: Union[str, callable] = 'mse', minPts: int = None, output: bool = False,
                 jaccard: bool = False, weighting: bool = False, exploitation_term: bool = False):
        """
        Params:
            data (pandas.DataFrame) - pandas dataframe with columns order 'object_id', 'time', 'cluster_id' containing cluster belongings,
                   features ..
                   Note: outliers should have negative labels/cluster_ids, these should be different for different times
        Optional:
            measure (str or callable) - for used quality measure, possible measures:
                                        'sse', 'mse', 'mae', 'max', 'dbi', 'exploit'
            minPts (int) - used minPts for density-based quality measure
            output (boolean) - whether intermediate results should be printed
            jaccard (boolean) - whether the jaccard index should be used for proportion
            weighting (boolean) - whether the weighting function should be used for subsequence_score
            exploitation_term (boolean) - whether the exploitation term should be included in CLOSE calculation
        """
        self._data = data
        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]
        self._cluster_column_name = self._column_names[2]

        self._jaccard = jaccard
        self._weighting = weighting
        self._exp_term = exploitation_term

        self._minPts = minPts
        self._output = output
        self.pos_measures = {### Measures for Clusters
                             'sse': self.calc_sse,  # NOTE: sse is not between 0 and 1
                             'mse': self.calc_mse,  # NOTE: mse is only between 0 and 1, if data is normalized
                             'mae': self.calc_mae,  # NOTE: mae is only between 0 and 1, if data is normalized
                             'max': self.calc_max_dist,
                             'dbi': self.calc_min_pts,
                             'None': self.return_zero,
                             ### Measures for Time Clusterings
                             'exploit': self.calc_exploit_at_t}

        if measure in self.pos_measures:
            self.measure = self.pos_measures[measure]
        elif callable(measure):
            self.measure = measure
        else:
            self.measure = self.pos_measures['mse']

    def rate_clustering(self, start_time: int = None, end_time: int = None, return_measures: bool = False) -> Union[float, dict]:
        """
        Optional:
            start_time (int) - time that should be considered as beginning
            end_time (int) - time which should be rated up to
            return_measures (boolean) - whether additional information such as average stability
                                                  and quality should be returned
        Returns:
            CLOSE score (float): rating of clustering regarding all clusters
            (dict): with key 'stability_evaluation', 'stability', 'quality', 'pre-factor' with additional information
                    if 'return_measures' is True
        """
        cluster_ratings = self.rate_clusters(start_time, end_time)
        gr_clusters = self._data.groupby(self._cluster_column_name)

        score = 0
        avg_quality = 0
        avg_stab = 0

        for cluster in cluster_ratings:
            cluster_objects = gr_clusters.get_group(cluster)[self._object_column_name].unique()
            cluster_time = gr_clusters.get_group(cluster)[self._time_column_name].iloc[0]
            feature_list = self.get_feature_list(cluster_objects, cluster_time)

            measure = self.measure(feature_list)
            avg_quality += measure
            avg_stab += cluster_ratings[cluster]
            score += (cluster_ratings[cluster] * (1 - measure))

        num_clusters = len(cluster_ratings)
        num_timestamps = self.get_num_timestamps(start_time, end_time)

        if num_clusters <= 0:
            if self._output:
             print('Clustering has no Clusters!!')
            return 0

        avg_quality /= num_clusters
        if self._output:
            print('Average Quality: ', str(avg_quality))
        avg_stab /= num_clusters
        if self._output:
            print('Average Stability: ', str(avg_stab))

        if self._exp_term:
            exp_term = self.calc_exploit()
            factor = (1 / num_clusters) * (1 - (num_timestamps / num_clusters) ** 2) * exp_term
        else:
            factor = (1 / num_clusters) * (1 - (num_timestamps / num_clusters)**2)

        if not return_measures:
            return score * factor

        else:
            return {'stability_evaluation': score * factor,
                    'stability': avg_stab,
                    'quality': avg_quality,
                    'pre-factor': (1 - (num_timestamps / num_clusters) ** 2)}

    def rate_time_clustering(self, start_time: int = None, end_time: int = None, return_measures: bool = False) -> Union[float, dict]:
        """
        Optional:
            start_time (optional) - int: time that should be considered as beginning
            end_time (optional) - int: time which should be rated up to
            return_measures (boolean) - whether additional information such as average stability and quality should be returned
        Returns:
            CLOSE score (float) - rating of clustering regarding all time clusterings
            (dict): with key 'stability_evaluation', 'stability', 'quality', 'pre-factor' with additional information
                    if 'return_measures' is True
        """
        cluster_ratings = self.rate_clusters(start_time, end_time)
        num_timestamps, timestamps = self.get_num_timestamps(start_time, end_time, return_timestamps=True)

        score = 0
        if return_measures:
            quality = 0
            stability = 0

        for time in timestamps:
            if not return_measures:
                score += self.calc_t_clustering_rating(cluster_ratings, time)
            else:
                cur_scores = self.calc_t_clustering_rating(cluster_ratings, time, return_measures=True)
                score += cur_scores['score']
                quality += cur_scores['quality']
                stability += cur_scores['stability']

        if return_measures:
            quality /= num_timestamps
            stability /= num_timestamps

        num_clusters = len(cluster_ratings)
        if num_clusters <= 0:
            if self._output:
             print('Over-Time Clustering has no Clusters!!')
            return 0

        if self._exp_term:
            exp_term = self.calc_exploit()
            factor = (1 / num_timestamps) * (1 - (num_timestamps / num_clusters) ** 2) * exp_term
        else:
            factor = (1 / num_timestamps) * (1 - (num_timestamps / num_clusters) ** 2)

        if not return_measures:
            return score * factor
        else:
            return {'stability_evaluation': score * factor,
                    'stability': stability,
                    'quality': quality,
                    'pre-factor': factor}

    def calc_t_clustering_rating(self, cluster_ratings: dict, time: int, return_measures: bool = False) -> Union[float, dict]:
        """
        Params:
            cluster_ratings (dict) - {<object_id>: <rating>} with ratings of objects
            time (int) - time that should be considered
        Optional:
            return_measures (boolean) - whether additional information such as average stability and quality should be returned
        Output:
            CLOSE score (float) - rating of clustering at considered time
            (dict): with key 'score', 'stability', 'quality' with additional information if 'return_measures' is True
        """
        avg_stab = 0

        clusters_at_time = self._data[self._data[self._time_column_name] == time][self._cluster_column_name].unique()
        clusters_at_time = np.delete(clusters_at_time, np.where(clusters_at_time < 0))
        
        for cluster in clusters_at_time:
            try:
                avg_stab += cluster_ratings[cluster]
            except:
                continue

        num_clusters = len(clusters_at_time)
        if num_clusters <= 0:
            if self._output:
             print('Time Clustering at Time ', str(time), ' has no Clusters!!')
            return 0

        avg_stab /= num_clusters
        if self._output:
            print('Average Stability at Time ', str(time), ' : ', str(avg_stab))

        quality = self.measure(time)
        if self._output:
            print('Quality of Clustering at Time ' , str(time), ' : ', str(quality))

        t_clustering_score = avg_stab * quality
        if not return_measures:
            return t_clustering_score
        else:
            return {
                'score': t_clustering_score,
                'stability': avg_stab,
                'quality': quality
            }

    def rate_clusters(self, start_time: int = None, end_time: int = None, id: Union[int, str, list] = None) -> dict:
        """
        Optional:
            start_time (int) - time that should be considered as beginning
            end_time (int) - time which should be rated up to
            id (int, str, list or None) - representing the cluster_ids that should be rated. If id is None,
                            all objects are rated
        Returns:
            ratings (dict) - {<cluster_id>: <rating>} with ratings of clusters
        """
        ids_to_rate = self.get_ids_to_rate(id, self._cluster_column_name, start_time, end_time)
        ids = ids_to_rate[:]

        # don't rate outliers
        for i in ids_to_rate:
            if int(i) < 0:
                ids.remove(i)

        ratings = self.calc_cluster_rating(ids, start_time)
        return ratings

    def calc_cluster_rating(self, ids_to_rate: Union[list, np.ndarray], start_time: int = None) -> dict:
        """
        Params:
            ids_to_rate (array-like) - list of clusters that should be rated
        Optional:
            start_time (int) - time that should be considered as beginning
        Returns:
            ratings - dict {<cluster_id>: <rating>} with ratings of clusters
        """
        if start_time is None:
            start_time = np.min(self._data[self._time_column_name].unique())

        ratings = {}
        cluster_compositions = self.obtain_cluster_compositions()
        gr_clusters = self._data.groupby(self._cluster_column_name)

        # iterate over all cluster ids
        for id in ids_to_rate:
            time = gr_clusters.get_group(id)[self._time_column_name].iloc[0]

            # rate the clusters of all timestamps except of the first one
            if time != start_time:
                num_merged_clusters = len(cluster_compositions[id])
                obj_list = gr_clusters.get_group(id)[self._object_column_name].unique().tolist()
                obj_ratings = self.calc_object_rating(cluster_compositions, obj_list, time)
                score = 0
                for obj in obj_ratings:
                    score += obj_ratings[obj]
                try:
                    score /= len(obj_ratings)
                except ZeroDivisionError:
                    if self._output:
                        print('Cluster ', str(id), ' has no non-outlier members.')
                    else:
                        continue

                clusters = list(cluster_compositions[id].keys())
                num_timestamps = len(self._data.loc[self._data[self._cluster_column_name].isin(clusters)]
                                     [self._time_column_name].unique())
                try:
                    div = num_merged_clusters / num_timestamps
                    score /= div
                except ZeroDivisionError:
                    if self._output:
                        print("<<ZeroDivisionError - Cluster Score>> Cluster ID: ", str(id), "  Merged Clusters: ", str(num_merged_clusters),
                         "  Num Timestamps: ", str(num_timestamps))
                    else:
                        continue
                ratings[id] = score

            # clusters of the first timestamp have a stability of 1.0
            else:
                ratings[id] = 1.0
        return ratings

    def rate_object(self, id: Union[int, str, list] = None, start_time: int = None, end_time: int = None) -> dict:
        """
        Optional:
            id (int, str, list or None) - representing the data points that should be rated. If id is None,
                            all objects are rated
            start_time (int) - time that should be considered as beginning
            end_time (int) - representing the timestamp which should be rated up to
        Returns:
            ratings (dict) - {<object_id>: <rating>} with ratings of objects
        """
        ids_to_rate = self.get_ids_to_rate(id, self._object_column_name)
        if end_time is None:
            end_time = np.max(self._data[self._time_column_name].unique())
        cluster_compositions = self.obtain_cluster_compositions()
        ratings = self.calc_object_rating(cluster_compositions, ids_to_rate, end_time, start_time)
        return ratings

    def calc_object_rating(self, cluster_composition: dict, ids_to_rate: Union[list, np.ndarray], end_time: int, start_time: int = None) -> dict:
        """
        Params:
            cluster_composition (dict) - {<cluster_id>: {<contained_cluster_id>: <proportion>}} containing the proportions of
                                  clusters (contained_cluster_id) that belong to cluster (cluster_id)
            ids_to_rate (array-like) - list of data points that should be rated
            end_time (int) - representing the timestamp which should be rated up to
        Optional:
            start_time (int) - time that should be considered as beginning
        Returns:
            ratings - dict {<object_id>: <rating>} with ratings of objects
        """
        ratings = {}
        gr_clusters = self._data.groupby(self._object_column_name)

        # iterate over object ids
        for id in ids_to_rate:
            cur_group = gr_clusters.get_group(id)
            cur_group = cur_group[cur_group[self._time_column_name] <= end_time]

            if start_time is not None:
                cur_group = cur_group[cur_group[self._time_column_name] >= start_time]

            try:
                # id of the cluster of the last considered timestamp
                last_cluster = cur_group[cur_group[self._time_column_name] == end_time][self._cluster_column_name].iloc[
                    0]
            except IndexError:
                print(">>INDEXERROR - LAST CLUSTER<< ID: ", str(id), ", Start Time: ", str(start_time), ", End Time: ",
                      str(end_time))
                continue

            # if object is an outlier for the considered timestamp, it is skipped
            if int(last_cluster) < 0:
                continue

            cluster_ids = cur_group[self._cluster_column_name].unique()

            object_ratings = []
            num_clusters = 0
            has_outlier = False
            for cluster in cluster_ids:
                if cluster == last_cluster:
                    continue
                # Add the proportion of clusters before last timestamp, that merged in last cluster
                else:
                    # outliers get worst rating of 0.0
                    if int(cluster) < 0:
                        object_ratings.append(0.0)
                        has_outlier = True
                    else:
                        object_ratings.append(cluster_composition[last_cluster][cluster])
                    num_clusters += 1
            if not has_outlier and len(object_ratings) == 0:
                # print(str(id) + " has no data before t=" + str(end_time))
                continue

            if self._weighting:
                try:
                    weighting_denominator = 0
                    for i in range(1, num_clusters + 1):
                        weighting_denominator += i

                    if num_clusters > 0:
                        object_rating = 0
                        for i in range(num_clusters):
                            object_rating += object_ratings[i] * ((i + 1) / weighting_denominator)

                    else:
                        continue
                except (TypeError, ZeroDivisionError):
                    # print(str(id) + " is not assigned to any cluster before t=" + str(end_time))
                    continue
            else:
                try:
                    object_rating = np.sum(object_ratings)
                    object_rating /= num_clusters
                except (TypeError, ZeroDivisionError):
                    # print(str(id) + " is not assigned to any cluster before t=" + str(end_time))
                    continue

            ratings[id] = round(object_rating, 3)
        return ratings

    def calc_exploit(self) -> float:
        """
        Returns:
            exploitation_term (float) - exploitation term for whole clustering
        """
        num_objects = len(self._data[self._object_column_name].unique())
        num_no_outliers = len(self._data[self._data[self._cluster_column_name] >= 0][self._object_column_name].unique())
        return num_no_outliers / num_objects


    ######## HELPER FUNCTIONS ########

    def get_feature_list(self, objects: Union[list, np.ndarray], time: int) -> np.ndarray:
        """
        Params:
            objects (array-like) - list of objects_ids that belong to considered cluster
            time (int) - time of cluster that is considered

        Output:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        """
        feature_list = []
        for obj in objects:
            features = self._data[
                (self._data[self._object_column_name] == obj) & (self._data[self._time_column_name] == time)]
            try:
                features = \
                    features.drop([self._object_column_name, self._cluster_column_name, self._time_column_name],
                                  axis=1).iloc[0].tolist()
            except IndexError:
                print(">>INDEXERROR - FEATURE LIST<< ID: ", str(obj), ", Time: ", str(time))
                continue

            if len(features) <= 0:
                print("No features found for object ", str(obj))
                continue
            feature_list.append(features)
        return np.array(feature_list)

    def get_num_timestamps(self, start_time: int, end_time: int, return_timestamps: bool = False) -> int:
        """
        Params:
            start_time (int) - first timestamp to be considered
            end_time (int) - last timestamp to be considered
        Optional:
            return_timestamps (boolean) - list of all timestamps
        Returns:
            num_timestamps (int) - number of timestamps between start_time and end_time
        """
        timestamp_list = self._data[self._time_column_name].unique()
        if start_time is not None:
            timestamp_list = [i for i in timestamp_list if i >= start_time]
        if end_time is not None:
            timestamp_list = [i for i in timestamp_list if i <= end_time]
        num_timestamps = len(timestamp_list)
        if not return_timestamps:
            return num_timestamps
        else:
            return num_timestamps, timestamp_list

    def get_ids_to_rate(self, id: Union[int, str, list], id_name: str, start_time: int = None, end_time: int = None) -> list:
        """
        Params:
            id (int, str, list or None) - representing the data points that should be rated. If id is None, all objects are rated
            id_name (str) - either self._cluster_column_name or self._object_column_name, which ids to extract
        Optional:
            start_time (int) - first timestamp to be considered
            end_time (int) - last timestamp to be considered
        Returns:
            ids_to_rate (list) - list of ids that should be rated
        """
        if id is None:
            data = self._data.copy()
            if start_time is not None:
                data = data[data[self._time_column_name] >= start_time]
            if end_time is not None:
                data = data[data[self._time_column_name] <= end_time]
            ids_to_rate = data[id_name].unique().tolist()
        elif isinstance(id, int) or isinstance(id, str):
            ids_to_rate = [id]
        elif isinstance(id, list):
            ids_to_rate = id[:]
        else:
            raise Exception('id has to be int, str, list or None')
        return ids_to_rate

    def obtain_cluster_compositions(self) -> dict:
        """
        Returns:
            cluster_compositions (dict) - dict of dicts {<cluster_id>: {<cluster_id>: <proportion>}} with cluster compositions

            Example:
                {5: {1: 1.0, 2: 0.1, 4: 0.5}} describes that
                        100% of cluster 1, 10% of cluster 2 and 50% of cluster 4 belong to cluster 5
        """
        cluster_compositions = {}
        g_clusters = self._data.groupby([self._time_column_name, self._cluster_column_name])

        if not self._jaccard:
            cluster_members = self._data.groupby(self._cluster_column_name).count()

        # iterate over all clusters - 'group' contains the time and cluster_id
        # and 'objects' is the corresponding dataframe
        for group, objects in g_clusters:
            # Ignore outliers
            if int(group[1]) < 0:
                continue

            objects = objects[self._object_column_name].values.tolist()

            # temporal intersection
            # select considered clusters with later timestamps than the current one to check which clusters the
            # current one merged into and count, how many objects of the current cluster are in the considered clusters
            # example of a series from the dataframe: [cluster_id, count] with [2, 10]
            # meaning: 10 objects of the current cluster merged into the cluster with the id 2
            temp_intersection = (self._data.loc[(self._data[self._object_column_name].isin(objects)) &
                            (self._data[self._time_column_name] > group[0])]).groupby(self._cluster_column_name).count()

            # iterate over all clusters which the current cluster has merged into
            # 'cluster' contains the cluster_id
            # and 'con_objects' is the corresponding number of objects of the temporal intersection
            for cluster, num_objects in temp_intersection.iterrows():
                # Ignore outliers
                if int(cluster) < 0:
                    continue

                # for all considered clusters save the proportion of the current cluster that merged into the considered
                # one
                # example: {3: {2: 0.3}, 4: {2: 0.1}}
                # meaning: 30% of (current) cluster 2 merged into (considered) cluster 3 and 10% into (considered) cluster 4
                if cluster not in cluster_compositions:
                    cluster_compositions[cluster] = {}

                if self._jaccard:
                    # cardinality of the union of both considered clusters
                    card_union = len(self._data.loc[(self._data[self._cluster_column_name] == cluster) |
                                                    (self._data[self._cluster_column_name] == group[1])]
                                     [self._object_column_name].unique())
                    # jaccard distance
                    cluster_compositions[cluster][group[1]] = round(float(num_objects.values[1]) /
                                                                    float(card_union), 3)
                else:
                    cluster_compositions[cluster][group[1]] = round(float(num_objects.values[1]) /
                                                                    float(cluster_members.loc[group[1]].values[1]), 3)
            if group[1] not in cluster_compositions:
                cluster_compositions[group[1]] = {}
        return cluster_compositions


    ######## QUALITY MEASURES ########

    @staticmethod
    def calc_sse(feature_list: list) -> float:
        """
        Params:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        Returns:
            sse (float) - sum of squared errors to centroid of cluster
        """
        centroid = np.average(feature_list, axis=0)
        sse = np.sum(np.power(feature_list - centroid[None, :], 2))
        return sse

    def calc_mse(self, feature_list: list) -> float:
        """
        Params:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        Returns:
            mse (float) - mean squared error of cluster
        """
        sse = self.calc_sse(feature_list)
        return sse / len(feature_list)

    @staticmethod
    def calc_mae(feature_list: list) -> float:
        """
        Params:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        Returns:
            mae (float) - mean average errors to centroid of cluster
        """
        centroid = np.average(feature_list, axis=0)
        mae = np.average(np.abs(feature_list - centroid[None, :]))
        return mae

    @staticmethod
    def calc_max_dist(feature_list: list) -> float:
        """
        Params:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        Returns:
            max_dist (float) - maximal distance of cluster member to centroid of cluster
        """
        max_dist = 0
        for i in range(len(feature_list) - 1):
            for j in range(i + 1, len(feature_list)):
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                if cur_dist > max_dist:
                    max_dist = cur_dist
        max_dist /= 2 ** (1 / 2)
        return max_dist

    def calc_min_pts(self, feature_list: list) -> float:
        """
        Params:
            feature_list (list) - list of lists containing the features of objects in the considered cluster
        Returns:
            avg_dist (float) - average distance of cluster members to their minPts neighbor
        """
        avg_dist = 0
        for i in range(len(feature_list)):
            dist_list = [10] * self._minPts
            for j in range(len(feature_list)):
                if i == j:
                    continue
                cur_dist = euclidean(np.array(feature_list[i]), np.array(feature_list[j]))
                for k in range(len(dist_list)):
                    if cur_dist < dist_list[k]:
                        dist_list.insert(k, cur_dist)
                        dist_list.pop(self._minPts)
            avg_dist += dist_list[self._minPts - 1]
        avg_dist /= len(feature_list)
        return avg_dist

    @staticmethod
    def return_zero():
        """
        Function is used if no quality measure should be used in CLOSE
        This is the case when only the exploitation term is considered

        Returns:
            0
        """
        return 0

    def calc_exploit_at_t(self, time: int) -> float:
        """
        Params:
            time (int) - time to be considered
        Returns:
            rating (float) - exploitation rating of time clustering
        """
        num_objects_at_t = len(self._data[self._data[self._time_column_name] == time][self._object_column_name].unique())
        num_no_outliers = len(self._data[(self._data[self._time_column_name] == time) &
                                         (self._data[self._cluster_column_name] >= 0)][self._object_column_name].unique())
        return num_no_outliers / num_objects_at_t


