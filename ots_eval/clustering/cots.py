import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import scipy.sparse.csgraph as csgraph
from typing import Union


class COTS(object):

    def __init__(self, data: pd.DataFrame, min_cf: float = 0.015, sw: int = 3):
        """
        Params:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features ..
                               Note: the object identifier should be in the first and the time
                                     in the second column of the DataFrame
        Optional:
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph (default: 0.015)
            sw (int) - width of sliding window (default: 3)
        """
        self._data = data
        self._min_cf = min_cf
        self._sw = sw

        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]

        self._timestamps = self._data[self._time_column_name].unique()
        self._timestamps.sort()

        self._object_ids = self._data[self._object_column_name].unique()
        self._object_ids.sort()

        self._similarities = np.array([])
        self._adaptabilities = np.array([])
        self._connection_factors = np.array([])
        self._temporal_connection_factors = np.array([])

    def get_factor_df(self, factor_type: str, **kwargs) -> pd.DataFrame:
        """
        Params:
            factor_type (str) - 'similarity', 'adaptability', 'connection', 'temporal_connection' or
                                'temporal_connection_sw' indicating the factor type to calculate
        Returns:
            factors (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID1', 'ObjectID2', 'Factor'
                                  Note: In case of 'adaptability' the DataFrame contains the columns
                                        'Time', 'ObjectID', 'Factor'
        """
        factors = self.get_factors(factor_type, **kwargs)
        return self.create_factors_df(factors)

    def get_factors(self, factor_type: str, **kwargs) -> np.ndarray:
        """
        Params:
            factor_type (str) - 'similarity', 'adaptability', 'connection', 'temporal_connection' or
                                'temporal_connection_sw' indicating the factor type to calculate
        Returns:
            factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the chosen
                              factors for all object pairs at all timestamps.
                              Note: In case of 'adaptability', the array has the shape (num_timestamps, num_objects)
        """
        if factor_type == 'similarity':
            return self.get_similarity()
        elif factor_type == 'adaptability':
            return self.get_adaptability()
        elif factor_type == 'connection':
            return self.get_connection_factor()
        elif factor_type == 'temporal_connection':
            return self.get_temporal_connection_factor()
        elif factor_type == 'temporal_connection_sw':
            return self.get_temporal_connection_factor_sw(**kwargs)
        else:
            print('Unknown Factor Type. Available Types are: "similarity", "adaptability", "connection", '
                  '"temporal_connection", and "temporal_connection_sw".')
            return None

    def get_similarity(self) -> np.ndarray:
        """
        Returns:
            similarities (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                   similarities for all object pairs at all timestamps.
                                   Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._similarities) <= 0:
            return self.calc_similarity(self.get_feature_vectors())
        else:
            return self._similarities

    def get_adaptability(self) -> np.ndarray:
        """
        Returns:
            adaptabilites (array) - array with shape (num_timestamps, num_objects) containing the adaptabilites for all
                                    objects at all timestamps.
                                    Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._adaptabilities) <= 0:
            return self.calc_adaptability(self.get_similarity())
        else:
            return self._adaptabilities

    def get_connection_factor(self) -> np.ndarray:
        """
        Returns:
            connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        connection factors for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._connection_factors) <= 0:
            return self.calc_connection_factor(self.get_similarity())
        else:
            return self._connection_factors

    def get_temporal_connection_factor(self) -> np.ndarray:
        """
        Returns:
            temporal_connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing
                                                  the temporal connection factors for all object pairs at all timestamps.
                                                  Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._temporal_connection_factors) <= 0:
            return self.calc_temporal_connection(self.get_connection_factor())
        else:
            return self._temporal_connection_factors

    def get_temporal_connection_factor_sw(self, sw: int = 3) -> np.ndarray:
        """
        Optional:
            sw (int) - width of sliding window (default: 3)
        Returns:
            temporal_connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing
                                                  the temporal connection factors for all object pairs at all timestamps
                                                  using a sliding window.
                                                  Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        if len(self._temporal_connection_factors) <= 0:
            return self.calc_temporal_connection_sw(self.get_connection_factor(), sw)
        else:
            return self._temporal_connection_factors

    def get_feature_vectors(self) -> np.ndarray:
        """
        Returns:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors for all objects at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        feature_vectors = []

        for time in self._timestamps:
            feature_vectors.append(self.get_feature_vectors_at_time(time))
        return np.array(feature_vectors)

    def get_feature_vectors_at_time(self, timestamp: int) -> np.ndarray:
        """
        Params:
            timestamp (int) - timestamp which the feature vectors should be extracted at
        Returns:
            feature_vector (array) - array with shape (num_objects, num_features) containing the feature vectors for
                                     all objects at the given timestamp.
                                     Note: The objectIDs are sorted ascending!
        """
        self._data = self._data.sort_values(by=[self._time_column_name, self._object_column_name])
        return self._data[self._data[self._time_column_name] == timestamp][self._column_names[2:]].values

    def get_distance_matrix(self) -> np.ndarray:
        """
        Returns:
            distance_matrix (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                      distances for all object pairs at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return self.calc_distance_matrix(self.get_feature_vectors())

    def get_clusters(self, min_cf: float, sw: int = 3) -> np.ndarray:
        """
        Params:
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph
        Optional:
            sw (int) - width of sliding window (default: 3)
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return self.calc_clusters(self.get_temporal_connection_factor_sw(sw), min_cf)

    def get_clusters_with_noise(self, min_cf: float, sw: int = 3) -> np.ndarray:
        """
        Params:
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph
        Optional:
            sw (int) - width of sliding window (default: 3)
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belongings including
                               noise of all objects at all timestamps.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return self.mark_outliers(self.calc_clusters(self.get_temporal_connection_factor_sw(sw), min_cf))

    def get_clusters_df(self, min_cf: float = 0.5, sw: int = 3) -> pd.DataFrame:
        """
        Optional:
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph (default:0.5)
            sw (int) - width of sliding window (default: 3)
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features.., 'cluster' containing the
                               the data and cluster belonging of all objects at all timestamps.
        """
        return self.add_cluster_to_df(self.get_clusters(min_cf, sw))

    def get_noisy_clusters_df(self, min_cf: float = 0.5, sw: int = 3) -> pd.DataFrame:
        """
        Optional:
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph (default:0.5)
            sw (int) - width of sliding window (default: 3)
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features.., 'cluster' containing the
                               the data and cluster belonging/noise of all objects at all timestamps.
        """
        return self.add_cluster_to_df(self.mark_outliers(self.get_clusters(min_cf, sw)))

    def create_clusters(self) -> pd.DataFrame:
        """
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'ObjectID', 'Time', features.., 'cluster' containing the
                               data and cluster belonging/noise of all objects at all timestamps.
        """
        return self.add_cluster_to_df(self.mark_outliers(self.get_clusters(self._min_cf, self._sw)))

    def create_factors_df(self, factors: Union[list, np.ndarray]) -> pd.DataFrame:
        """
        Params:
            factors (array-like) - containing the factors matrix with dimensions
                                   (num_timestamps, num_objects, num_objects) with factors[i, j, k] being
                                   the factors of object j for object k at time i.
                                   Note: The timestamps as well as the objectIDs must be sorted ascending!
                                   Note2: In case of 'adaptability' the array has the shape (num_timestamps, num_objects)
        Returns:
            factors (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID1', 'ObjectID2', 'Factor'
                                  Note: In case of 'adaptability' the DataFrame contains the columns
                                        'Time', 'ObjectID', 'Factor'
        """
        if len(factors.shape) == 3:
            facts = pd.DataFrame(columns=['Time', 'ObjectID1', 'ObjectID2', 'Factor'])

            for time in range(len(self._timestamps)):
                for i in range(len(self._object_ids)):
                    for j in range(len(self._object_ids)):
                        facts = facts.append({'Time': self._timestamps[time],
                                              'ObjectID1': self._object_ids[i],
                                              'ObjectID2': self._object_ids[j],
                                              'Factor': factors[time, i, j]}, ignore_index=True)
        elif len(factors.shape) == 2:
            facts = pd.DataFrame(columns=['Time', 'ObjectID', 'Factor'])

            for time in range(len(self._timestamps)):
                for i in range(len(self._object_ids)):
                    facts = facts.append({'Time': self._timestamps[time],
                                          'ObjectID': self._object_ids[i],
                                          'Factor': factors[time, i]}, ignore_index=True)
        else:
            return None
        return facts

    def add_cluster_to_df(self, clusters: Union[list, np.ndarray]) -> pd.DataFrame:
        """
        Params:
            clusters (array-like) - containing the clusters matrix with dimensions
                                       (num_timestamps, num_objects) with clusters[i, j] being
                                       the cluster belonging of object j at time i.
                                       Note: The timestamps as well as the objectIDs must be sorted ascending!
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'Time', 'ObjectID', features.., 'cluster'
        """
        self._data = self._data.assign(cluster=-1)

        for time in range(len(self._timestamps)):
            for oid in range(len(self._object_ids)):
                self._data.loc[(self._data[self._time_column_name] == self._timestamps[time]) &
                               (self._data[self._object_column_name] == self._object_ids[oid]),
                               'cluster'] = clusters[time][oid]
        return self._data

    @staticmethod
    def calc_distance_matrix_at_time(feature_vectors: np.ndarray) -> np.ndarray:
        """
        Params:
            feature_vectors (array) - array with shape (num_objects, num_features) containing the feature vectors for
                                      all objects at a given timestamp
        Returns:
            distance_matrix (array) - array with shape (num_objects, num_objects) containing the distances for
                                     all object pairs at the given timestamp.
                                     Note: The objectIDs are sorted ascending!
        """
        return distance_matrix(feature_vectors, feature_vectors)

    def calc_distance_matrix(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Params:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors for all objects at all timestamps
        Returns:
            distance_matrix (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                      distances for all object pairs at all timestamps.
                                      Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        dist_matrix = []
        for time in range(len(feature_vectors)):
            dist_matrix.append(self.calc_distance_matrix_at_time(feature_vectors[time]))
        return np.array(dist_matrix)

    def calc_similarity(self, feature_vectors: np.ndarray) -> np.ndarray:
        """
        Params:
            feature_vectors (array) - array with shape (num_timestamps, num_objects, num_features) containing the
                                      feature vectors of all objects at all timestamps
        Returns:
            similarities (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        similarities  for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        distances = self.calc_distance_matrix(feature_vectors)
        try:
            distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        except ValueError:
            print('Data must not have missing values!')
            raise
        self._similarities = (1 - distances)**2
        return self._similarities

    @staticmethod
    def calc_adaptability(similarities: np.ndarray) -> np.ndarray:
        """
        Params:
            similarities (array) - array with shape (num_objects, num_timestamps, num_objects) containing the
                                  similarities of all objects at all timestamps
        Returns:
            adaptabilies (array) - array with shape (num_timestamps, num_objects) containing the
                                   adaptabilites for all objects at all timestamps.
                                   Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        adaptabilities = (np.nansum(similarities, axis=0) - 1) / (np.count_nonzero(~np.isnan(similarities), axis=0) - 1)
        return adaptabilities

    def calc_connection_factor(self, similarities: np.ndarray) -> np.ndarray:
        """
        Params:
            similarities (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                  similarities of all objects at all timestamps
        Returns:
            connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                        connection factors for all object pairs at all timestamps.
                                        Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        similarities = np.moveaxis(similarities, -2, 0)
        adaptabilities = self.calc_adaptability(similarities)
        connection_factors = similarities * adaptabilities
        self._connection_factors = np.moveaxis(connection_factors, 0, -2)
        return self._connection_factors

    def calc_temporal_connection(self, connection_factors: np.ndarray) -> np.ndarray:
        """
        Params:
            connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                         connection factors of all objects at all timestamps
        Returns:
            temporal_connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing
                                                  the temporal connection factors for all object pairs at all timestamps.
                                                  Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        temp_connection_factors = np.zeros(connection_factors.shape)
        sum_pref = np.sum(connection_factors, axis=0)

        for time in range(len(connection_factors)):
            for i in range(len(connection_factors[time])):
                for j in range(len(connection_factors[time, i])):
                    avg_pref = (sum_pref[i, j] - connection_factors[time, i, j]) / \
                               (len(connection_factors) - 1)
                    if not np.isnan(avg_pref) and avg_pref > 0:
                        temp_connection_factors[time, i, j] = (connection_factors[time, i, j] + avg_pref) / 2
                    else:
                        temp_connection_factors[time, i, j] = connection_factors[time, i, j]
        self._temporal_connection_factors = temp_connection_factors
        return temp_connection_factors

    def calc_temporal_connection_sw(self, connection_factors: np.ndarray, sw: int = 3) -> np.ndarray:
        """
        Params:
            connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                  connection factors of all objects at all timestamps
        Optional:
            sw (int) - width of sliding window, default:3
        Returns:
            temp_connection_factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the
                                              temporal connection factors for all object pairs at all timestamps using a
                                              sliding window.
                                              Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        temp_connection_factors = np.zeros(connection_factors.shape)

        for time in range(len(connection_factors)):

            prec = round((sw - 1) / 2) + 1
            if prec > time:
                prec = time

            suc = int((sw - 1) / 2)
            if suc > connection_factors.shape[0] - time - 1:
                suc = connection_factors.shape[0] - time - 1

            lsw = prec + suc

            for i in range(len(connection_factors[time])):
                for j in range(len(connection_factors[time, i])):
                    avg_pref = (np.nansum(connection_factors[time - prec:time + suc + 1, i, j]) - connection_factors[time, i, j]) / lsw
                    if not np.isnan(avg_pref) and avg_pref > 0:
                        temp_connection_factors[time, i, j] = (connection_factors[time, i, j] + avg_pref) / 2
                    else:
                        temp_connection_factors[time, i, j] = connection_factors[time, i, j]
        self._temporal_connection_factors = temp_connection_factors
        return temp_connection_factors

    @staticmethod
    def calc_cluster_peers_indices(factors: np.ndarray, min_cf: float) -> np.ndarray:
        """
        Params:
            factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the factors
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph
        Returns:
            indices (array) - array with shape (num_edges, 3) containing the lists [time, object1, object2] indicating
                              edges of the graph
        """
        return np.argwhere(factors >= min_cf)

    @staticmethod
    def calc_cluster_peers(factors: np.ndarray, min_cf: float) -> np.ndarray:
        """
        Params:
            factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the factors
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects, num_objets) with clusters[i, j, k] = True
                               indicating an edge between objects j and k at timestamp i.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        return factors >= min_cf

    def calc_clusters(self, factors: np.ndarray, min_cf: float) -> np.ndarray:
        """
        Params:
            factors (array) - array with shape (num_timestamps, num_objects, num_objects) containing the factors
                                  for all object pairs at all timestamps.
                                  Note: The timestamps as well as the objectIDs must be sorted ascending!
            min_cf (float) - threshold for the minimum connection factor for inserting edges to the graph
        Returns:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        clusters = []
        for time in range(len(self._timestamps)):
            peers = self.calc_cluster_peers(factors[time], min_cf)
            graph = csgraph.csgraph_from_dense(peers)
            n_components, labels = csgraph.connected_components(csgraph=graph, directed=False, return_labels=True)
            clusters.append(labels)
        return np.array(clusters)

    def mark_outliers(self, clusters: np.ndarray) -> np.ndarray:
        """
        Params:
            clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belongings of all
                               objects for all timestamps.
                               Note: The timestamps as well as the objectIDs must be sorted ascending!
        Returns:
            new_clusters (array) - array with shape (num_timestamps, num_objects) containing the cluster belonging of all
                               objects at all timestamps, whereby cluster with only one element are marked as noise.
                               Note: The timestamps as well as the objectIDs are sorted ascending!
        """
        new_clusters = []
        for time in range(len(self._timestamps)):
            time_clusters = clusters[time]
            hist, _ = np.histogram(time_clusters, bins=np.arange(np.max(time_clusters)+2))
            outlier_clusters = np.argwhere(hist == 1).flatten()
            time_clusters = [x if x not in outlier_clusters else -1 for x in time_clusters ]
            new_clusters.append(np.array(time_clusters))
        return np.array(new_clusters)