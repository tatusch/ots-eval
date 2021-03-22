import pandas as pd
import numpy as np
from typing import Tuple


class DACT(object):

    def __init__(self, data: pd.DataFrame):
        """
        Params:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id' containing objects,
                               timestamps, cluster belongings, features ..
                               Note: The first three columns can have custom names as long as they represent the object
                               identifier, the timestamp and the cluster identifier in the right order
        """
        self._data = data.astype({data.columns.values[0]: str})
        self._column_names = data.columns.values
        self._object_column_name = self._column_names[0]
        self._time_column_name = self._column_names[1]
        self._cluster_column_name = self._column_names[2]
        self._object_ids = self._data[self._object_column_name].unique()
        self._num_objects = len(self._object_ids)
        self._memberships = self.calc_membership_matrices()

        self._outlier_rating = None
        self._outlier_result = None

    def get_outliers(self, tau: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Parameters:
            tau (float) - threshold for outlier detection

        Returns:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id', 'outlier'
            outlier_result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating', 'distance' and 'outlier'
        """
        self.calc_outlier_degree()
        return self.mark_outliers(tau)

    def calc_membership_matrices(self) -> dict:
        """
        Returns:
            memberships (dict) - {<timestamp>: <membership_matrix>} containing the membership matrices for all timestamps
        """
        memberships = {}
        timestamps = self._data[self._time_column_name].unique()

        for i in range(len(timestamps)):
            relevant_data = self._data[self._data[self._time_column_name] == timestamps[i]]
            memberships[timestamps[i]] = np.zeros((self._num_objects, self._num_objects))
            cluster_ids = relevant_data[self._cluster_column_name].unique()
            for cid in cluster_ids:
                if cid >= 0:
                    members = relevant_data[relevant_data[self._cluster_column_name] == cid][self._object_column_name].unique()
                    for j in range(len(members)-1):
                        index_j = np.argwhere(self._object_ids == members[j])[0][0]
                        for k in range(j+1, len(members)):
                            index_k = np.argwhere(self._object_ids == members[k])[0][0]
                            memberships[timestamps[i]][index_j][index_k] = 1
                            memberships[timestamps[i]][index_k][index_j] = 1
        return memberships

    def calc_cohesion_matrix(self, start_time: int, end_time: int) -> np.ndarray:
        """
        Params:
            start_time (int) - time that should be considered as beginning
            end_time (int) - int representing the timestamp which should be rated up to
        Returns:
            cohesion_matrix (array) - array of shape (num_objects, num_objects) containing the cohesion values of all time
                                      series to each other for the considered time period
        """
        timestamps = self._data[self._time_column_name].unique()
        timestamps = timestamps[np.where(timestamps >= start_time)]
        timestamps = timestamps[np.where(timestamps <= end_time)]
        cohesion_matrix = np.zeros((self._num_objects, self._num_objects))
        for time in timestamps:
            cohesion_matrix = cohesion_matrix + self._memberships[time]
        return cohesion_matrix

    def calc_subsequence_ratings(self, start_time: int, end_time: int) -> pd.DataFrame:
        """
        Params:
            start_time (int) - time that should be considered as beginning
            end_time (int) - int representing the timestamp which should be rated up to
        Returns:
            subsequence_ratings (pandas.DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time', 'rating'
        """
        subsequence_ratings = pd.DataFrame(columns=[self._object_column_name, 'start_time', 'end_time', 'rating'])
        cohesion_matrix = self.calc_cohesion_matrix(start_time, end_time)

        for i in range(self._num_objects):
            summed = np.sum(cohesion_matrix[i])
            non_zero = np.count_nonzero(cohesion_matrix[i])
            num_timestamps = len(self._data[(self._data[self._object_column_name] == self._object_ids[i]) &
                                            (self._data[self._time_column_name] >= start_time) &
                                            (self._data[self._time_column_name] <= end_time)])

            if non_zero > 0 and num_timestamps > 1:
                rating = summed / non_zero
                rating = rating / num_timestamps
            elif num_timestamps <= 1:
                rating = -1
            else:
                rating = 0

            subsequence_ratings = subsequence_ratings.append({self._object_column_name: str(self._object_ids[i]),
                                                                  'start_time': start_time,
                                                                  'end_time': end_time,
                                                                  'rating': rating}, ignore_index=True, sort=False)
        return subsequence_ratings

    def calc_tau(self, results: pd.DataFrame, percent: float) -> float:
        """
        Params:
            results (pandas.DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time', 'rating', 'distance'
            percent (float) - percentage which amount of the data is expected to be outliers (0.5 represents 50%)
        Returns:
            max_outscore (float) - threshold parameter tau in order to detect the right amount of outliers
        """
        scores = results['distance'].values
        scores = np.sort(scores)
        scores = scores[::-1]
        unique_scores = np.unique(scores)
        max_outscore = scores[round(percent * len(scores)) - 1]
        while np.count_nonzero(scores >= max_outscore) > round(percent * len(scores)):
            try:
                score_index = np.where(unique_scores == max_outscore)[0][0] + 1
                max_outscore = unique_scores[score_index]
            except IndexError:
                break
        return max_outscore

    def calc_outlier_degree(self) -> pd.DataFrame:
        """
        Returns:
            outlier_rating (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time', 'rating', 'distance'
        """
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()

        self._outlier_rating = pd.DataFrame(
            columns=[self._object_column_name, 'start_time', 'end_time', 'rating', 'distance'])

        for t1 in range(len(timestamps) - 1):
            for t2 in range(t1 + 1, len(timestamps)):
                subsequence_ratings = self.calc_subsequence_ratings(timestamps[t1], timestamps[t2])
                subsequence_ratings = subsequence_ratings.assign(distance=0)
                relevant_data = self._data[self._data[self._time_column_name] == timestamps[t2]]
                for cid in relevant_data[self._cluster_column_name].unique():
                    if cid >= 0:
                        relevant_ids = relevant_data[relevant_data[self._cluster_column_name] == cid][self._object_column_name].unique()
                        best_score = np.amax(subsequence_ratings[subsequence_ratings[self._object_column_name].isin(relevant_ids)]['rating'].values)
                        subsequence_ratings[subsequence_ratings[self._object_column_name].isin(relevant_ids)] = \
                            subsequence_ratings[subsequence_ratings[self._object_column_name].isin(relevant_ids)].assign(distance=lambda x: best_score - x['rating'])

                self._outlier_rating = self._outlier_rating.append(subsequence_ratings)

        noise_object_ids = self._data[self._data[self._cluster_column_name] < 0][self._object_column_name].unique()
        for object_id in noise_object_ids:
            noisy_timestamps = self._data[(self._data[self._object_column_name] == object_id) &
                                          (self._data[self._cluster_column_name] < 0)][self._time_column_name].unique().tolist()
            noisy_timestamps.sort()
            start_index = np.where(timestamps == noisy_timestamps[0])[0][0]
            counter = 1
            for i in range(1, len(noisy_timestamps)):
                if noisy_timestamps[i] == timestamps[start_index + counter]:
                    counter += 1
                else:
                    if counter > 1:
                        self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                            'start_time': timestamps[start_index],
                                                                            'end_time': timestamps[start_index + counter - 1],
                                                                            'rating': -1,
                                                                            'distance': -1}, ignore_index=True)
                    start_index = np.where(timestamps == noisy_timestamps[i])[0][0]
                    counter = 1
            if counter > 1:
                self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                    'start_time': timestamps[start_index],
                                                                    'end_time': timestamps[start_index + counter -1],
                                                                    'rating': -1,
                                                                    'distance': -1}, ignore_index=True)

        return self._outlier_rating

    def mark_outliers(self, tau: float = None, percent: float = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optional:
            tau (float) - threshold for outlier detection
            percent (float) - percentage which amount of the data is expected to be outliers (0.5 represents 50%)
                              If percentage is not None, tau is calculated in order to find the right amount of outliers
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id', 'outlier'
            outlier_result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating', 'distance' and 'outlier'
        """
        if percent is not None:
            used_tau = self.calc_tau(self._outlier_rating, percent)
        elif tau:
            used_tau = tau
        else:
            used_tau = 0.4
        self._outlier_result = self._outlier_rating[((self._outlier_rating['distance'] >= used_tau) &
                                                     (self._outlier_rating['distance'] <= 1)) |
                                                    (self._outlier_rating['distance'] == -1)]

        time_points = self._data[self._time_column_name].unique().tolist()
        time_points.sort()

        self._data = self._data.assign(outlier=1)
        self._data = self._data.astype({self._object_column_name: str})
        self._outlier_result = self._outlier_result.astype({self._object_column_name: str})

        for index, row in self._outlier_result.iterrows():
            for time_point in time_points[time_points.index(int(row['start_time'])): time_points.index(
                    int(row['end_time'])) + 1]:
                if row['distance'] == -1:
                    self._data.loc[(self._data[self._time_column_name] == time_point) & (
                            self._data[self._object_column_name] == row[self._object_column_name]), 'outlier'] = -2
                else:
                    self._data.loc[(self._data[self._time_column_name] == time_point) & (
                            self._data[self._object_column_name] == row[self._object_column_name]), 'outlier'] = -1

        return self._data, self._outlier_result

    def calc_statistic_cluster_outliers(self, factor: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optional:
            factor (float) - prefactor for stddev to detect statistic outliers
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id', 'outlier'
            outlier_result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating', 'distance' and 'outlier'
        """
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()

        self._outlier_rating = pd.DataFrame(
            columns=[self._object_column_name, 'start_time', 'end_time', 'rating', 'distance', 'outlier'])

        for t1 in range(len(timestamps) - 1):
            for t2 in range(t1 + 1, len(timestamps)):
                subsequence_ratings = self.calc_subsequence_ratings(timestamps[t1], timestamps[t2])
                subsequence_ratings = subsequence_ratings.assign(distance=0)
                subsequence_ratings = subsequence_ratings.assign(outlier=0)
                relevant_data = self._data[self._data[self._time_column_name] == timestamps[t2]]
                for cid in relevant_data[self._cluster_column_name].unique():
                    if cid >= 0:
                        relevant_ids = relevant_data[relevant_data[self._cluster_column_name] == cid][
                            self._object_column_name].unique()
                        best_score = np.amax(
                            subsequence_ratings[subsequence_ratings[self._object_column_name].isin(relevant_ids)][
                                'rating'].values)
                        subsequence_ratings[subsequence_ratings[self._object_column_name].isin(relevant_ids)] = \
                            subsequence_ratings[
                                subsequence_ratings[self._object_column_name].isin(relevant_ids)].assign(
                                distance=lambda x: best_score - x['rating'])
                        squared_distances = subsequence_ratings[(subsequence_ratings[self._object_column_name].isin(relevant_ids)) &
                                                                (subsequence_ratings['distance'] <= 1)]['distance'].values ** 2
                        std_dev = (np.sum(squared_distances) / len(squared_distances)) ** 0.5
                        subsequence_ratings[(subsequence_ratings[self._object_column_name].isin(relevant_ids)) &
                                                                (subsequence_ratings['distance'] <= 1)] = \
                            subsequence_ratings[
                                (subsequence_ratings[self._object_column_name].isin(relevant_ids)) &
                                                                (subsequence_ratings['distance'] <= 1)].assign(
                                outlier=lambda x: (x['distance'] <= factor * std_dev) - 1)

                self._outlier_rating = self._outlier_rating.append(subsequence_ratings)

        noise_object_ids = self._data[self._data[self._cluster_column_name] < 0][self._object_column_name].unique()
        for object_id in noise_object_ids:
            noisy_timestamps = self._data[(self._data[self._object_column_name] == object_id) &
                                          (self._data[self._cluster_column_name] < 0)][
                self._time_column_name].unique().tolist()
            noisy_timestamps.sort()
            start_index = np.where(timestamps == noisy_timestamps[0])[0][0]
            counter = 1
            for i in range(1, len(noisy_timestamps)):
                if noisy_timestamps[i] == timestamps[start_index + counter]:
                    counter += 1
                else:
                    if counter > 1:
                        self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                            'start_time': timestamps[start_index],
                                                                            'end_time': timestamps[
                                                                                start_index + counter - 1],
                                                                            'rating': -1,
                                                                            'distance': -1,
                                                                            'outlier': -2}, ignore_index=True)
                    start_index = np.where(timestamps == noisy_timestamps[i])[0][0]
                    counter = 1
            if counter > 1:
                self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                    'start_time': timestamps[start_index],
                                                                    'end_time': timestamps[start_index + counter - 1],
                                                                    'rating': -1,
                                                                    'distance': -1,
                                                                    'outlier': -2}, ignore_index=True)
        self._outlier_result = self._outlier_rating[self._outlier_rating['outlier'] < 0]
        timestamps = timestamps.tolist()
        self._data = self._data.assign(outlier=0)
        for index, row in self._outlier_result.iterrows():
            for time_point in timestamps[timestamps.index(int(row['start_time'])): timestamps.index(
                    int(row['end_time'])) + 1]:
                self._data.loc[(self._data[self._time_column_name] == time_point) & (
                        self._data[self._object_column_name] == row[self._object_column_name]), 'outlier'] = row['outlier']

        return self._data, self._outlier_result

    def calc_statistic_outliers(self, factor: float = 3.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optional:
            factor (float) - prefactor for stddev to detect statistic outliers
        Returns:
            data (DataFrame) - pandas DataFrame with columns 'object_id', 'time', 'cluster_id', 'outlier'
            outlier_result (DataFrame) - pandas DataFrame with columns 'object_id', 'start_time', 'end_time',
                                         'cluster_end_time', 'rating', 'distance' and 'outlier'
        """
        timestamps = self._data[self._time_column_name].unique()
        timestamps.sort()

        self._outlier_rating = pd.DataFrame(
            columns=[self._object_column_name, 'start_time', 'end_time', 'rating', 'distance', 'outlier'])

        for t1 in range(len(timestamps) - 1):
            for t2 in range(t1 + 1, len(timestamps)):
                subsequence_ratings = self.calc_subsequence_ratings(timestamps[t1], timestamps[t2])
                subsequence_ratings = subsequence_ratings.assign(distance=0)
                subsequence_ratings = subsequence_ratings.assign(outlier=0)
                std_dev = np.std(subsequence_ratings[subsequence_ratings['rating'] >= 0]['rating'].values)
                mean = np.mean(subsequence_ratings[subsequence_ratings['rating'] >= 0]['rating'].values)
                subsequence_ratings = subsequence_ratings.assign(distance=lambda x: abs(x['rating'] - mean))
                subsequence_ratings[subsequence_ratings['rating'] >= 0] = subsequence_ratings[subsequence_ratings['rating'] >= 0].assign(outlier=lambda x: (x['distance'] <= factor * std_dev) - 1)
                self._outlier_rating = self._outlier_rating.append(subsequence_ratings)

        noise_object_ids = self._data[self._data[self._cluster_column_name] < 0][self._object_column_name].unique()
        for object_id in noise_object_ids:
            noisy_timestamps = self._data[(self._data[self._object_column_name] == object_id) &
                                          (self._data[self._cluster_column_name] < 0)][
                self._time_column_name].unique().tolist()
            noisy_timestamps.sort()
            start_index = np.where(timestamps == noisy_timestamps[0])[0][0]
            counter = 1
            for i in range(1, len(noisy_timestamps)):
                if noisy_timestamps[i] == timestamps[start_index + counter]:
                    counter += 1
                else:
                    if counter > 1:
                        self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                            'start_time': timestamps[start_index],
                                                                            'end_time': timestamps[
                                                                                start_index + counter - 1],
                                                                            'rating': -1,
                                                                            'distance': -1,
                                                                            'outlier': -2}, ignore_index=True)
                    start_index = np.where(timestamps == noisy_timestamps[i])[0][0]
                    counter = 1
            if counter > 1:
                self._outlier_rating = self._outlier_rating.append({self._object_column_name: object_id,
                                                                    'start_time': timestamps[start_index],
                                                                    'end_time': timestamps[start_index + counter - 1],
                                                                    'rating': -1,
                                                                    'distance': -1,
                                                                    'outlier': -2}, ignore_index=True)
        self._outlier_result = self._outlier_rating[self._outlier_rating['outlier'] < 0]
        timestamps = timestamps.tolist()
        self._data = self._data.assign(outlier=0)
        for index, row in self._outlier_result.iterrows():
            for time_point in timestamps[timestamps.index(int(row['start_time'])): timestamps.index(
                    int(row['end_time'])) + 1]:
                self._data.loc[(self._data[self._time_column_name] == time_point) & (
                        self._data[self._object_column_name] == row[self._object_column_name]), 'outlier'] = row['outlier']

        return self._data, self._outlier_result



