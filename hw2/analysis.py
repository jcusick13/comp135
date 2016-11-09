# analysis.py

from nb import *
import math
import matplotlib.pyplot as plt


def learning_curve(dataset, results1, max_index1, max_acc1,
                   results2, max_index2, max_acc2,
                   results3, max_index3, max_acc3,
                   results4, max_index4, max_acc4):
    """Creates test set accuracy plot showing
    test set accuracy as a function of k, training set size.

    dataset: str, name of dataset to plot
    results: list, accuracy for each classification run,
                orderded by increasing k
    max_index: int, index of acc for greatest accuracy
                from test set run
    max_acc: float, greatest accuracy from test set run
    """

    # Plot max accuracy point, accuracy of all runs in results

    # Type 1, m = 0
    plt.plot([max_index1], [max_acc1], 'ob')
    plt.plot([x / 10.0 for x in range(1, 11)], [y for y in results1], 'b',
             label='Type 1 max. accuracy (m=0): %f, N = %i'
             % (max_acc1, max_index1))

    # Type 1, m=1
    plt.plot([max_index2], [max_acc2], 'or')
    plt.plot([x / 10.0 for x in range(1, 11)], [y for y in results2], 'r',
             label='Type 1 max. accuracy (m=1): %f, N = %i'
             % (max_acc2, max_index2))

    # Type 2, m=0
    plt.plot([max_index3], [max_acc3], 'og')
    plt.plot([x / 10.0 for x in range(1, 11)], [y for y in results3], 'g',
             label='Type 2 max. accuracy (m=0): %f, N = %i'
             % (max_acc3, max_index3))

    # Type 2, m=1
    plt.plot([max_index4], [max_acc4], 'oc')
    plt.plot([x / 10.0 for x in range(1, 11)], [y for y in results4], 'c',
             label='Type 2 max. accuracy (m=1): %f, N = %i'
             % (max_acc4, max_index4))

    # Create axis/title labels and legend
    plt.xlabel('N')
    plt.ylabel('Test set accuracy')
    plt.title(dataset + ' Classification Evaluation')
    plt.legend(loc='lower left', fontsize='small', numpoints=1)

    # Align axes, show grid
    plt.axis([0.1, 1, 0.0, 1])
    plt.grid(True)


def learning_curve_smoothing(dataset, results1, max_index1, max_acc1,
                             results2, max_index2, max_acc2, m):
    """Creates test set accuracy plot showing
    test set accuracy as a function of m, smoothing parameter
    for two datasets (Type1 and Type2).

    dataset: str, name of dataset to plot
    results: list, accuracy for each classification run,
                orderded by increasing m
    max_index: int, index of acc from greatest accuracy
                from test set run
    max_acc: float, greatest accuracy from test set run
    m: list, full set of m values that were used for evaluation
    """

    # Plot max accuracy point, accuracy of all runs in results
    plt.plot([max_index1], [max_acc1], 'ob')
    plt.plot([x for x in m], [y for y in results1],
             label='Type 1 max. accuracy: %f, m = %i' % (max_acc1, max_index1))
    plt.plot([max_index2], [max_acc2], 'og')
    plt.plot([x for x in m], [y for y in results2],
             label='Type 2 max. accuracy: %f, m = %i' % (max_acc2, max_index2))

    # Create axis/title labels and legend
    plt.xlabel('m')
    plt.ylabel('Test set accuracy')
    plt.xscale('log')
    plt.title(dataset + ' Classification Evaluation (Smoothing)')
    plt.legend(loc='lower left', fontsize='small', numpoints=1)

    # Align axes, show grid
    plt.axes([0.0, 10.0, 0.0, 1.0])
    plt.grid(True)


if __name__ == '__main__':

    # Establish datasets
    comp = r'pp2data/ibmmac/'
    sport = r'pp2data/sport/'

    #
    # Eval. 1 - Initial accuracy evaluations
    #

    # Use to record results of all runs
    comp_t1_m0 = []
    comp_t1_m1 = []
    comp_t2_m0 = []
    comp_t2_m1 = []

    sport_t1_m0 = []
    sport_t1_m1 = []
    sport_t2_m0 = []
    sport_t2_m1 = []

    # Create breaks for training size sets (0.1N, 0.2N, ..., N)
    breaks = [x / 10.0 for x in range(1, 11)]
    tot_lines = 76

    # Gather results as a function of train set size
    for i in breaks:
        print 'Training Naive Bayes on %f of data...' % (i)
        brk = math.ceil(i * tot_lines)

        # Computer Hardware
        # m = 0
        prior1, cp, bag, prior2 = learn_nb(comp + 'index_train', 0, brk)
        comp_t1_m0.append(naive_bayes(prior1, cp, bag,
                                      comp + 'index_test', 'Type1'))
        comp_t2_m0.append(naive_bayes(prior2, cp, bag,
                                      comp + 'index_test', 'Type2'))

        # m = 1
        prior1, cp, bag, prior2, = learn_nb(comp + 'index_train', 1, brk)
        comp_t1_m1.append(naive_bayes(prior1, cp, bag,
                                      comp + 'index_test', 'Type1'))
        comp_t2_m1.append(naive_bayes(prior2, cp, bag,
                                      comp + 'index_test', 'Type2'))

        # Sports
        # m = 0
        prior1, cp, bag, prior2 = learn_nb(sport + 'index_train', 0, brk)
        sport_t1_m0.append(naive_bayes(prior1, cp, bag,
                                       sport + 'index_test', 'Type1'))
        sport_t2_m0.append(naive_bayes(prior2, cp, bag,
                                       sport + 'index_test', 'Type2'))

        # m = 1
        prior1, cp, bag, prior2 = learn_nb(sport + 'index_train', 1, brk)
        sport_t1_m1.append(naive_bayes(prior1, cp, bag,
                                       sport + 'index_test', 'Type1'))
        sport_t2_m1.append(naive_bayes(prior2, cp, bag,
                                       sport + 'index_test', 'Type2'))

    all_results = [['Computer Hardware', comp_t1_m0, comp_t1_m1,
                    comp_t2_m0, comp_t2_m1],
                   ['Sports', sport_t1_m0, sport_t1_m1,
                    sport_t2_m0, sport_t2_m1]]

    print 'Plotting accuracy results for Section 1...'
    for res in all_results:
        # Record values at best accuracy
        max_acc1 = max(res[1])
        max_index1 = breaks[res[1].index(max_acc1)]

        max_acc2 = max(res[2])
        max_index2 = breaks[res[2].index(max_acc2)]

        max_acc3 = max(res[3])
        max_index3 = breaks[res[3].index(max_acc3)]

        max_acc4 = max(res[4])
        max_index4 = breaks[res[4].index(max_acc4)]

        # Create output plot
        fig = plt.figure()
        learning_curve(res[0], res[1], max_index1, max_acc1,
                       res[2], max_index2, max_acc2,
                       res[3], max_index3, max_acc3,
                       res[4], max_index4, max_acc4)
        fig.savefig('%s_overall.gif' % (res[0]))
        del fig

    #
    # Eval. 2 - Accuracy as a function of m (smoothing)
    #

    m = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
         1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Use to record results of all m runs
    comp_t1_res = []
    comp_t2_res = []
    sport_t1_res = []
    sport_t2_res = []

    for i in range(len(m)):
        print 'Running smoothing where m =', m[i]
        # Computer Hardware
        prior1, cp, bag, prior2 = learn_nb(comp + 'index_train', i)
        comp_t1_res.append(naive_bayes(prior1, cp, bag,
                                       comp + 'index_test', 'Type1'))
        comp_t2_res.append(naive_bayes(prior2, cp, bag,
                                       comp + 'index_test', 'Type2'))

        # Sports
        prior1, cp, bag, prior2 = learn_nb(sport + 'index_train', i)
        sport_t1_res.append(naive_bayes(prior1, cp, bag,
                                        sport + 'index_test', 'Type1'))
        sport_t2_res.append(naive_bayes(prior2, cp, bag,
                                        sport + 'index_test', 'Type2'))

    all_results = [['Computer Hardware', comp_t1_res, comp_t2_res],
                   ['Sports', sport_t1_res, sport_t2_res]]

    print 'Plotting accuracy results for Section 2...'
    for res in all_results:
        # Record values at best accuracy
        max_acc1 = max(res[1])
        max_index1 = m[res[1].index(max_acc1)]
        max_acc2 = max(res[2])
        max_index2 = m[res[2].index(max_acc2)]

        # Create output plot
        fig = plt.figure()
        learning_curve_smoothing(res[0], res[1], max_index1, max_acc1,
                                 res[2], max_index2, max_acc2, m)
        fig.savefig('%s_smoothing_acc.gif' % (res[0]))
        del fig
