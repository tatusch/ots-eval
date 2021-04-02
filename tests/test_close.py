import pandas
from testtatusch.stability_evaluation.close import CLOSE


def test():
    test_data = [[1, 1, 1, 1 / 3, 1 / 6], [2, 1, 1, 2 / 3, 1 / 6], [3, 1, 1, 1 / 3, 2 / 6], [4, 1, 2, 2 / 3, 4 / 6],
                 [5, 1, 2, 3 / 3, 4 / 6], [6, 1, 2, 2 / 3, 5 / 6],
                 [1, 2, 3, 2 / 3, 1 / 6], [2, 2, 3, 3 / 3, 1 / 6], [3, 2, 3, 2 / 3, 2 / 6], [4, 2, 4, 2 / 3, 5 / 6],
                 [5, 2, 4, 3 / 3, 5 / 6], [6, 2, 4, 2 / 3, 6 / 6],
                 [1, 3, 5, 2 / 3, 1 / 6], [2, 3, 5, 2 / 3, 2 / 6], [3, 3, 5, 1 / 3, 1 / 6], [4, 3, 6, 2 / 3, 5 / 6],
                 [5, 3, 6, 3 / 3, 4 / 6], [6, 3, 6, 1 / 3, 6 / 6]]

    data = pandas.DataFrame(test_data, columns=['object_id', 'time', 'cluster_id', 'feature1', 'feature2'])
    rater = CLOSE(data, 'mae', 2, jaccard=True, weighting=True, exploitation_term=True)
    rater.rate_clustering()

    rater = CLOSE(data, 'exploit', 2, jaccard=True, weighting=True)
    rater.rate_time_clustering()


def main():
    try:
        test()
        print('CLOSE \t \t \t \t Ok.')
    except:
        print('CLOSE \t \t \t \t Failed.')


main()