import pandas as pd
from testtatusch.outlier_detection.dact import DACT


def test():
    test_data = pd.DataFrame(data={'ObjectID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                       'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                                       'Cluster': [1, 3, 5, 1, 3, 5, 1, 4, 5, 2, 4, 6, 2, 4, 6, 2, 3, -1]})
    outlier_detector = DACT(test_data)

    outlier_detector.get_outliers(tau=0.3)
    rating = outlier_detector.calc_outlier_degree()

    tau = outlier_detector.calc_tau(rating, percent=0.1)
    outlier_detector.mark_outliers(tau=tau)

    outlier_detector.calc_statistic_outliers(factor=1.5)


def main():
    try:
        test()
        print('DACT \t \t \t \t OK.')
    except:
        print('DACT \t \t \t \t Failed.')


main()