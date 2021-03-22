from .cots import COTS
import pandas as pd


test_data = pd.DataFrame({'ObjectID': ['a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f',
                                           'a', 'b', 'c', 'd', 'e', 'f'],
                              'Time': [1, 1, 1, 1, 1, 1,
                                       2, 2, 2, 2, 2, 2,
                                       3, 3, 3, 3, 3, 3,
                                       4, 4, 4, 4, 4, 4,
                                       5, 5, 5, 5, 5, 5],
                              'Feature': [0.9, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.3, 0.2, 0.1,
                                          0.8, 0.6, 0.5, 0.4, 0.2, 0.1,
                                          0.9, 0.6, 0.5, 0.4, 0.3, 0.1,
                                          0.9, 0.5, 0.6, 0.4, 0.2, 0.1]})


def example_cots():
    cots = COTS(test_data)

    clusters = cots.get_clusters(min_cf=0.2, sw=3)
    print("Calculated Clusters as Numpy Array with Sliding Window Size 3: ")
    print(clusters)
    print()

    clusters_df = cots.get_clusters_df(min_cf=0.2, sw=3)
    print("Calculated Clusters as Pandas DataFrame with Sliding Window Size 3: ")
    print(clusters_df)
    print()

    tmp_cf = cots.get_temporal_connection_factor()
    print("Calculated Temporal Connection Factors: ")
    print(tmp_cf)
    print()

    cf = cots.get_factors(factor_type="connection")
    print("Calculated Connection Factors with General Function: ")
    print(cf)
    print()


if __name__ == '__main__':
    example_cots()