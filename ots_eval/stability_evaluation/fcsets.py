import numpy as np
from collections import OrderedDict
from typing import Union
import math
from numba import jit


class FCSETS(object):

    def __init__(self, data: dict):
        """
           Params:
               data (dict) - with format {<timestamp>: <membership_matrix>}, with the membership matrices being numpy.arrays of shape (num_clusters, num_time_series)
        """
        self.data = data
        timestamps = self.data.keys()
        self.start_year = min(timestamps)
        self.end_year = max(timestamps)

    def rate_clustering(self, start_time: int = None, end_time: int = None) -> float:
        """
        Optional:
            start_time (int) - time that should be considered as beginning
            end_time (int) - time which should be rated up to
        Returns:
            FCSETS score (float): rating of clustering regarding all clusters
        """
        stabilities = self.calc_sequence_stability(start_time, end_time)
        fcsets_score = sum(stabilities.values()) / len(stabilities.values())
        return fcsets_score

    def calc_sequence_stability(self, start_time: int = None, end_time: int = None) -> dict:
        """
        Optional:
            start_time (int) - time that should be considered as beginning
            end_time (int) - time which should be rated up to
        Returns:
            stability_dict (dict): {<object_id>: <stability_score>} containing the stability values of all time series
        """
        if start_time is None:
            start_time = self.start_year
        if end_time is None:
            end_time = self.end_year

        num_time_series = self.data[self.start_year].shape[1]
        hr_dict = self.calculate_rel_ass_agreement(start_time, end_time, num_time_series)

        # calculate out of place distance for every time series
        stability_dict = dict()
        for i in range(0, num_time_series):
            normalize_counter = 0
            oop_sum = 0
            for y1 in range(start_time, end_time):
                for y2 in range(start_time, end_time):
                    if y2 > y1:
                        oop_sum = oop_sum + self.weighted_oop_distance(hr_dict[y1][i], hr_dict[y2][i])
                        normalize_counter = normalize_counter + 1
            stability_dict[i] = 1 - (oop_sum / normalize_counter)
        return stability_dict

    def calculate_rel_ass_agreement(self, start_time: int, end_time: int, num_time_series: int) -> dict:
        """
        Params:
            start_time (int) - time that should be considered as beginning
            end_time (int) - time which should be rated up to
            num_time_series (int) - how many time series are considered
        Returns:
            hr_dict (dict) - {<timestamp>: {<object_id>: {<object_id>: <hr_index>}}} containing the Hüllermeier-Rifqi indices
                             of all time series to each other
        """
        hr_dict = dict()

        # Calculate e_p for every pair of time series
        for i in range(start_time, end_time + 1):
            hr_dict[i] = dict()
            for j in range(0, num_time_series):
                hr_dict[i][j] = OrderedDict()
                for k in range(0, num_time_series):
                    hr_dict[i][j][k] = self.e_p(self.data[i][:, j], self.data[i][:, k])
                # Sort dicts by e_p
                sorted_dict = OrderedDict(
                    {k: v for k, v in sorted(hr_dict[i][j].items(), key=lambda item: item[1], reverse=True)})
                hr_dict[i][j] = sorted_dict
        return hr_dict

    @staticmethod
    def weighted_oop_distance(a: OrderedDict, b: OrderedDict) -> float:
        """
        Params:
            a (OrderedDict) - {<object_id>: <hr_index>} containing the Hüllermeier-Rifqi indices regarding a time series to all others in the first timestamp
            b (OrderedDict) - {<object_id>: <hr_index>} containing the Hüllermeier-Rifqi indices regarding the same time series to all others in the second timestamp
        Returns:
            weighted_oop_distance (float) - weighted out of place distance of two dicts (regarding their keys)
        """
        distance = 0
        e_p_sum = 0
        for k in a.keys():
            distance = distance + a[k] * math.fabs(list(a.keys()).index(k) - list(b.keys()).index(k))
            e_p_sum += a[k]
        return distance / (e_p_sum * len(list(a.keys())))

    @staticmethod
    @jit
    def e_p(u_x: Union[list, np.array], u_y: Union[list, np.array]) -> float:
        """
        Params:
            u_x (array_like) - membership matrix of first timestamp
            u_y (array_like) - membership matrix of second timestamp
        Returns:
            e_p (float) - equivalence relation between the two timestamps
        """
        ep = 0
        for i in range(0, u_x.shape[0]):
            ep = ep + np.absolute(u_x[i] - u_y[i])
        return 1 - ep/2
