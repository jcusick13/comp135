# analysis.py
from arff_reader import *
from knn import *
import matplotlib.pyplot as plt


def accuracy_plot(dataset, knn_res, max_index, max_acc, j48_res):
    """Creates test set accuracy plot comparing
    kNN accuracy as a function of k and a
    J48 decision tree.

    dataset: str name of dataset to plot
    knn_res: list of accuracy from kNN runs
    max_index: int of k at greatest kNN accuracy
    max_acc: float of greatest accuracy from kNN runs
    j48_res: float of J48 accuracy
    """

    # Plot max accuracy of kNN, kNN accuracy, J48 accuracy
    plt.plot([max_index], [max_acc], 'ob')
    plt.plot([x for x in range(1, 26)], results,
             label='kNN (max accuracy: %f, k = %i)' % (max_acc, max_index))
    plt.plot([x for x in range(1, 26)], [j48_res for x in range(1, 26)],
             label='J48 (accuracy: %f)' % (j48_res))

    # Create axis/title labels and legend
    plt.xlabel('k')
    plt.ylabel('Test set accuracy')
    plt.title(dataset + ' Classification Evaluation')
    plt.legend(loc='lower left', fontsize='small', numpoints=1)

    # Align axes, show grid
    plt.axis([1, 25, 0.5, 1])
    plt.grid(True)


if __name__ == '__main__':

    # Build dictionary of input files for kNN and
    # accuracy of already ran J48 classification
    ds = {}

    # Ionosphere
    iono_train = arff('ionosphere_train.arff')
    iono_test = arff('ionosphere_test.arff')
    ds['Ionosphere'] = [iono_train, iono_test, 0.91453]

    # Irrelevant
    irr_train = arff('irrelevant_train.arff')
    irr_test = arff('irrelevant_test.arff')
    ds['Irrelevant'] = [irr_train, irr_test, 0.645]

    # Mfeat-Fourier
    m_train = arff('mfeat-fourier_train.arff')
    m_test = arff('mfeat-fourier_test.arff')
    ds['Mfeat-Fourier'] = [m_train, m_test, 0.746627]

    # Spambase
    spam_train = arff('spambase_train.arff')
    spam_test = arff('spambase_test.arff')
    ds['Spambase'] = [spam_train, spam_test, 0.915906]

    # REMOVE ME !
    # test = {}
    # test['Ionosphere_test'] = [iono_train, iono_test, 0.91453]

    #
    # 3.4 Evaluating kNN with respect to k
    #

    for key in test:
        print 'Running analysis for %s...' % (str(key))

        # Run kNN algo to get accuracy as function of k
        results = [knn(ds[key][0].data, ds[key][1].data, k)
                   for k in range(1, 26)]
        # Record values at best accuracy
        max_acc = max(results)
        max_index = results.index(max_acc) + 1

        # Create output plot
        fig = plt.figure()
        accuracy_plot(str(key), results, max_index, max_acc, ds[key][2])
        fig.savefig('%s.gif' % (str(key)))
        del fig
