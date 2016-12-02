# arff_reader.py
from collections import namedtuple
import keyword


class Arff:
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
        self.classes = {}

        # Read attributes into field_names
        with open(self.name, 'r') as f:

            # Dict of string replacements for symbols to avoid
            avoid = {';': 'semicn',
                     '(': 'lparen',
                     '[': 'lbrack',
                     '!': 'excmk',
                     '$': 'dollar',
                     '#': 'hash'}

            for line in f:
                lsplit = line.split(' ')
                if lsplit[0].lower() == '@attribute':
                    # Protect against forbidden column names within
                    # a named tuple

                    # Avoid python keywords by adding underscore
                    if keyword.iskeyword(lsplit[1]):
                        self.field_names.append(lsplit[1] + '_')

                    # Avoid non-alphanumeric using lookup dict 'avoid'
                    elif (lsplit[1][-1] in avoid):
                        self.field_names.append(lsplit[1][:-1] +
                                                avoid[lsplit[1][-1]])

                    # Avoid starting field with a number
                    elif self.is_numeric(lsplit[1][0]):
                        self.field_names.append('attr' + lsplit[1].lower())

                    # Attribute name safe to use as is
                    else:
                        self.field_names.append(line.split(' ')[1])

                elif lsplit[0].lower() == '@data':
                    break

        # Use field_names to define named tuple holder for each @data row
        Row = namedtuple('Row', self.field_names)

        # Read through raw input file
        with open(self.name, 'r') as f:

            # Skip over header/attribute info
            for line in f:
                if line.strip().lower() == '@data':
                    break

            # Arrive at first row of @data, append each row
            # as a namedtuple to self.data
            for l in f:
                # Create list, strip \n, convert all but last to float
                in_vals = l.rstrip().split(',')
                flt_in_vals = [float(i) for i in in_vals[:-1]]
                flt_in_vals.append(in_vals[-1])

                # Update classification count dictionary
                # First time seen, create entry in dict
                if in_vals[-1] not in self.classes:
                    self.classes[in_vals[-1]] = 1

                # Already seen classification, increase count
                else:
                    self.classes[in_vals[-1]] += 1

                # Add namedtuple of input @data row to self.data
                self.data.append(Row._make(flt_in_vals))

    def is_numeric(self, s):
        """Tests if input string is a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False

    def pprint(self, row):
        """Formatted printing of the entry at the
        provided row.
        """
        print '\n ------'
        for i in self.field_names:
            print i + ': ', getattr(row, i)
