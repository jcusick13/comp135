# decision_trees.py
import math


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


def gain(collection, attr):
    """Accepts list of counts for each
    classification class, 2-d list of classifications
    after split with attribute attr.
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
