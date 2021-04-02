import pandas
import numpy
from .close import CLOSE
from .fcsets import FCSETS


def rate_clustering(data):
    rater = CLOSE(data, 'mae', 2, jaccard=True, weighting=True, exploitation_term=True)
    return rater.rate_clustering()


def rate_time_clusterings(data):
    rater = CLOSE(data, 'exploit', 2, jaccard=True, weighting=True)
    return rater.rate_time_clustering()


def rate_fuzzy_clustering(data):
    rater = FCSETS(data)
    return rater.rate_clustering()


def example_close():
    # test_data for 6 time series with 3 timestamps, 2 clusters per timestamp and 2 features (features are optional as those are not considered in CLOSE)
    test_data = [[1, 1, 1, 1 / 3, 1 / 6], [2, 1, 1, 2 / 3, 1 / 6], [3, 1, 1, 1 / 3, 2 / 6], [4, 1, 2, 2 / 3, 4 / 6], [5, 1, 2, 3 / 3, 4 / 6], [6, 1, 2, 2 / 3, 5 / 6],
                 [1, 2, 3, 2 / 3, 1 / 6], [2, 2, 3, 3 / 3, 1 / 6], [3, 2, 3, 2 / 3, 2 / 6], [4, 2, 4, 2 / 3, 5 / 6], [5, 2, 4, 3 / 3, 5 / 6], [6, 2, 4, 2 / 3, 6 / 6],
                 [1, 3, 5, 2 / 3, 1 / 6], [2, 3, 5, 2 / 3, 2 / 6], [3, 3, 5, 1 / 3, 1 / 6], [4, 3, 6, 2 / 3, 5 / 6], [5, 3, 6, 3 / 3, 4 / 6], [6, 3, 6, 1 / 3, 6 / 6]]

    data = pandas.DataFrame(test_data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])

    clustering_score = rate_clustering(data)
    print('Total CLOSE Score with CLUSTER quality measure: ', str(clustering_score))

    t_clustering_score = rate_time_clusterings(data)
    print('Total CLOSE Score with CLUSTERING quality measure: ', str(t_clustering_score))
    return


def example_fcsets():
    # test_data for 5 time series with 4 timestamps and 2 clusters per year
    # dict with {<timestamp>: <membership matrix with shape (num_clusters, num_time_series)>}
    test_data = {
        1: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        2: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        3: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        4: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]])
    }

    clustering_score = rate_fuzzy_clustering(test_data)
    print('Total FCSETS Score for perfectly stable clustering: ', str(clustering_score))

    # test_data for 5 time series with 4 timestamps and 2 clusters per year
    test_data = {
        1: numpy.array([[0.999, 0.9905, 0.01, 0.002, 0.028],
                        [0.001, 0.0005, 0.99, 0.998, 0.972]]),
        2: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        3: numpy.array([[0.999, 0.9905, 0.01, 0.002, 0.028],
                        [0.001, 0.0005, 0.99, 0.998, 0.972]]),
        4: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]])
    }
    clustering_score = rate_fuzzy_clustering(test_data)
    print('Total FCSETS Score with 1/5 transitioning time series: ', str(clustering_score))
    return


if __name__ == '__main__':
    example_close()
    example_fcsets()