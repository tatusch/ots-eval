from testtatusch.clustering.cots import COTS
import pandas as pd


def test():
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
    cots = COTS(test_data)
    cots.get_clusters(min_cf=0.2, sw=3)
    cots.get_clusters_df(min_cf=0.2, sw=3)
    cots.get_temporal_connection_factor()
    cots.get_factors(factor_type="connection")


def main():
    try:
        test()
        print('COTS \t \t \t \t OK.')
    except:
        print('COTS \t \t \t \t Failed.')


main()