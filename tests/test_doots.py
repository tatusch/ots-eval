import pandas as pd
from testtatusch.outlier_detection.doots import DOOTS


def test():
    test_data = pd.DataFrame(data={'ObjectID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                   'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                                   'Cluster': [1, 3, 5, 1, 3, 5, 1, 4, 5, 2, 4, 6, 2, 4, 6, 2, 3, -1]})
    outlier_detector = DOOTS(test_data, weighting=False, jaccard=False)

    outlier_detector.calc_outlier_degree()

    outlier_detector.mark_outliers(tau=0.3)

    outlier_detector.get_outliers(tau=0.3)


def main():
    try:
        test()
        print('DOOTS \t \t \t \t OK.')
    except:
        print('DOOTS \t \t \t \t Failed.')


main()