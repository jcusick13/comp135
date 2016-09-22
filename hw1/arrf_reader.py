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


def euclidean_dist(x, y, weight=1):
    """Calculates Euclidean distance between x and y,
    where x and y are both tuples.
    Accepts weights as floating point numbers.
    """
    for i in range(len(x) - 1):
        print x[i]


iono = arrf('ionosphere_test.arff')
euclidean_dist(iono.data[0], iono.data[1])
