# knn.py
import math


def discrete_counts(dataset, feature):
    """Splits dataset into 5 groups, based on an ordered
    list of values from feature. Converts the numerical
    value of the feature into one of 5 discrete categories,
    corresponding to the five groups. Returns a dictionary of
    discrete category: list of counts per output classification
    i.e. {0: [16, 8], 1: [7, 17], 2: [8, 15], 3: [4, 19], 4: [6, 17]}
    when there are only two output classifications.

    This is intended to be used as the input for measuring the
    information gain of the selected feature in the dataset.

    dataset: arff class
    feature: string name of feature
    """

    # Create list of feature value, classification for each
    # entry in dataset
    info = [[getattr(row, feature), row[-1]] for row in dataset.data]
    print info[:5]

    # Sort list by feature values
    info.sort(lambda x, y: cmp(x[0], y[0]))
    print info[:5]
    print dataset.classes

    # Prepare to divide info list into list of 5 groups
    grp_size = len(info) // 5
    groups = []

    # Create and append 5 groups of exact same size
    for i in range(0, grp_size * 5, grp_size):
        groups.append(info[i:i + grp_size])

    # Evenly distribute leftover rows, if applicable
    if len(info) % 5 != 0:
        grp_assigned = 0
        for i in range(grp_size * 5, len(info)):
            groups[grp_assigned].append(info[i])
            grp_assigned += 1

    # Create dictionary of list of classification counts by group
    grp_counts = {}
    number = 0

    for grp in groups:
        # Array used for index locations of each output classification
        class_idx_ref = [i for i in dataset.classes]
        # Array of counter totals for each output classification
        class_count = [0 for i in dataset.classes]

        for exp in grp:
            # Increase index of list by one each time that index's
            # classification is seen
            class_count[class_idx_ref.index(exp[-1])] += 1

        grp_counts[number] = class_count
        number += 1

    return grp_counts


def entropy(collection):
    """Accepts list of counts for each
    classification class, returns entropy
    as expected encoding length in bits.
    """

    # Get sum of classified query instances
    s = float(reduce(lambda a, b: a + b, collection))

    # Calculate individual entropy terms using proportion
    # of classification i relative to the full collection
    entropy = [-((i / s) * math.log(i / s, 2)) for i in collection]

    # Sum entropy over each classification
    return reduce(lambda a, b: a + b, entropy)


def euclidean_dist(x, y, weight=1.0):
    """Calculates Euclidean distance between x and y,
    where x and y are both tuples.
    Accepts weights as floating point numbers.
    """
    sq_diff = 0
    for i in range(len(x)):
        sq_diff += weight * (x[i] - y[i]) ** 2

    return sq_diff ** (1 / 2.0)


def find_neighbors(training, inst, k):
    """Performs a linear search through the
    list of tuples in training and returns list of
    the k neighbors nearest to the instance tuple inst
    using a euclidean distance search. List is orderded
    from smallest to largest distance.

    Does not consider the distance of the last element
    in each tuple as it is assumed to be a classification,
    not numeric value.

    Biased for keeping first neighbor seen in case of
    a tie in distance between two potential nearest
    neighbors.
    """

    nn = []

    # Evaluate distance of each training example
    for exp in training:
        dist = euclidean_dist(exp[:-1], inst)

        if len(nn) < k:
            # Naively add example to nn list with its Euclid
            # distance added as first tuple element
            nn.append((dist,) + exp)
            nn.sort(key=lambda t: t[0])

        else:
            # Compare dist to training examples currently in nn
            # starting with worst example
            for nbr in reversed(nn):
                if dist < nbr[0]:
                    # Remove worst neighbor, add exp, reorder nn
                    del nn[-1]
                    nn.append((dist,) + exp)
                    nn.sort(key=lambda t: t[0])
                    break

    # Remove stored euclidean distances
    nn_cleaned = []
    for i in nn:
        nn_cleaned.append(i[1:])

    return nn_cleaned


def info_gain(collection, attr):
    """Accepts list of counts for each
    classification class, 2-d list of classifications
    after split with attribute attr. Returns information
    gain (decrease in entropy) in bits.
    """

    # Get sum of classified query instances
    s = float(reduce(lambda a, b: a + b, collection))

    # Calculate sum of proportional entropy after
    # split using attr
    postsplit = 0.0
    count = 0
    for value in attr:
        sv = reduce(lambda a, b: a + b, attr[count])
        # Weighted entropy of each class after split
        postsplit += (sv / s) * entropy(attr[count])
        count += 1

    return entropy(collection) - postsplit


def majority_count(examples):
    """Returns the majority classification count from a list
    of tuples. Each tuple is a data point and the last
    element is that point's classification.

    In case of a tie, the first classification of the tied
    elements seen will be returned.
    """
    counts = {}
    for exp in examples:

        # First time seen, add entry to dict
        if exp[-1] not in counts:
            counts[exp[-1]] = 1

        # Already seen classification, increase count
        else:
            counts[exp[-1]] += 1

    return max(counts, key=lambda k: counts[k])


def knn(train, test, k):
    """For each instance in test, obtains a classification
    for it by taking the majority vote of the k nearest
    neighbors (by Euclidean distances) from the full
    set of instances in train.

    Returns test set classification accuracy.
    """
    acc = {'correct': 0.0, 'wrong': 0.0}

    for exp in test:

        # Find k-nearest neighbors
        nn = find_neighbors(train, exp, k)

        # Get classification of k-neighbors, compare to
        # actual classification of example
        if majority_count(nn) == exp[-1]:
            acc['correct'] += 1
        else:
            acc['wrong'] += 1

    return acc['correct'] / sum(acc.values())


def knn_info_gain(train, test, k, n):
    """Performs an initial information gain analysis
    on each of the attributes of the train dataset and
    allows the caller to pass n, where n is the number
    of attributes with the highest information gain to
    be used when assigning examples from the test dataset
    to a classification.

    Returns test set classification accuracy.
    """

    # Calculate info gain for each attribute in train
