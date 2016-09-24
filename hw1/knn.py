# knn.py


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
