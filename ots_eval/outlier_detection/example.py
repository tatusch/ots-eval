import pandas as pd
from .doots import DOOTS
from .dact import DACT


test_data = pd.DataFrame(data={'ObjectID': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                   'Time': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                                'Cluster': [1, 3, 5, 1, 3, 5, 1, 4, 5, 2, 4, 6, 2, 4, 6, 2, 3, -1]})

def example_doots():
    outlier_detector = DOOTS(test_data, weighting=False, jaccard=False)

    rating = outlier_detector.calc_outlier_degree()
    print("Subsequence Scores for all Subsequences with DOOTS: ")
    print(rating)

    data, outliers = outlier_detector.mark_outliers(tau=0.3)
    print("Detected Outliers with DOOTS and tau=0.3: ")
    print(outliers)

    data, detected_outliers = outlier_detector.get_outliers(tau=0.3)
    print("Immediately Calculated Outliers with DOOTS and tau=0.3: ")
    print(detected_outliers)


def example_dact():
    outlier_detector = DACT(test_data)

    data, detected_outliers = outlier_detector.get_outliers(tau=0.3)
    print("Immediately Calculated Outliers with DACT and tau=0.3: ")
    print(detected_outliers)

    rating = outlier_detector.calc_outlier_degree()
    print("Subsequence Scores for all Subsequences with DACT: ")
    print(rating)

    tau = outlier_detector.calc_tau(rating, percent=0.1)
    print("Calculated tau in order to detect 5% of the data as outliers: ")
    print(tau)
    data, outliers = outlier_detector.mark_outliers(tau=tau)
    print("Detected Outliers with DACT and tau=", str(tau), ": ")
    print(outliers)

    data, detected_outliers = outlier_detector.calc_statistic_outliers(factor=1.5)
    print("Calculated Statistic Outliers with DACT and 3*stdv: ")
    print(detected_outliers)


if __name__ == '__main__':
    example_doots()
    example_dact()
