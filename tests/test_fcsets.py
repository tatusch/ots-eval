import numpy
from testtatusch.stability_evaluation.fcsets import FCSETS


def test():
    data = {
        1: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        2: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        3: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]]),
        4: numpy.array([[0.999, 0.9905, 0.989, 0.002, 0.028],
                        [0.001, 0.0005, 0.011, 0.998, 0.972]])
    }
    rater = FCSETS(data)
    rater.rate_clustering()

def main():
    try:
        test()
        print('FCSETS \t \t \t \t Ok.')
    except:
        print('FCSETS \t \t \t \t Failed.')


main()