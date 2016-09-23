# arrf_reader.py
from collections import namedtuple
import keyword


class arrf:
    """Container for .arff dataset allowing
    for manulipation in python. Assumes all features
    features are numerical except last column
    which contains any number of discrete values.

    Values are stored as named tuples, with all
    numerical categories as floats, and a discrete
    classification category as string.
    """

    def __init__(self, infile):
        self.name = infile
        self.field_names = []
        self.data = []
        self.classification = []

        # Read attributes into field_names
        with open(self.name, 'r') as f:
            for line in f:
                lsplit = line.split(' ')
                if lsplit[0] == '@attribute':
                    if keyword.iskeyword(lsplit[1]):
                        self.field_names.append(lsplit[1] + '_')
                    else:
                        self.field_names.append(line.split(' ')[1])
                elif lsplit[0] == '@data':
                    break

        # Use field_names to define named tuple holder for each @data row
        Row = namedtuple('Row', self.field_names)

        # Read through raw input file
        with open(self.name, 'r') as f:

            # Skip over header/attribute info
            for line in f:
                if line.strip() == '@data':
                    break

            # Arrive at first row of @Data, append each row
            # as a namedtuple to self.data
            for l in f:
                # Create list, strip \n, convert all but last to float
                in_vals = l.rstrip().split(',')
                flt_in_vals = [float(i) for i in in_vals[:-1]]
                flt_in_vals.append(in_vals[-1])

                self.data.append(Row._make(flt_in_vals))

    def pprint(self, row):
        """Formatted printing of the entry at the
        provided row.
        """
        print '\n ------'
        for i in self.field_names:
            print i + ': ', getattr(row, i)


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

    Biased for keeping first neighbor seen in case of
    a tie in distance between two potential nearest
    neighbors.
    """

    nn = []

    # Evaluate distance of each training example
    for exp in training:
        dist = euclidean_dist(exp, inst)

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


if __name__ == '__main__':

    # iono = arrf('ionosphere_test.arff')

    train = [(1, 2), (2, 3), (10, 10), (12, 12)]
    instance = (2, 2)

    print find_neighbors(train, instance, 2)
