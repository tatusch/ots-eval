import numpy as np
from typing import Union
import math


class FCSETS(object):

    def __init__(self, data: dict):
        """
           Params:
               data (dict) - with format {<timestamp>: <membership_matrix>}, with the membership matrices being
                             numpy.arrays of shape (num_clusters, num_time_series)
        """
        self.data = data
        timestamps = self.data.keys()
        self.num_timestamps = len(timestamps)
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

        # calculate stability(T_l) for every time series l
        stability_dict = dict()
        for l in range(0, num_time_series):
            stability_dict[l] = 0
            for i in range(start_time, end_time):
                for r in range(i + 1, end_time + 1):
                    counter = 0
                    denominator = 0
                    for s in range(0, num_time_series):
                        counter = counter \
                                  + math.pow(hr_dict[i][l][s], num_time_series) \
                                  * math.pow(math.fabs(hr_dict[i][l][s] - hr_dict[r][l][s]), 2)
                        denominator = denominator + math.pow(hr_dict[i][l][s], num_time_series)
                    stability_dict[l] = stability_dict[l] + (counter / denominator)
            stability_dict[l] = 1 - (2 / (self.num_timestamps * (self.num_timestamps - 1))) * stability_dict[l]
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
                hr_dict[i][j] = dict()
                for k in range(0, num_time_series):
                    hr_dict[i][j][k] = self.e_p(self.data[i][:, j], self.data[i][:, k])
        return hr_dict

    @staticmethod
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
        return 1 - ep / 2