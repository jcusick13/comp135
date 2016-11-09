# ann_test.py
import unittest
from ann import *


class TestAnnFunctions(unittest.TestCase):

    #
    # sigmoid()
    #

    def test_sigmoid_convert_int(self):
        """Ensures output is the same regardless if input
        is given as int or float.
        """

        self.assertEqual(sigmoid(5), sigmoid(5.0))

    #
    # sigmoid_p()
    #

    def test_sigmoid_p_output(self):
        """Ensures correct output of sigmoid derivative."""

        self.assertEqual(round(sigmoid_p(0.993), 3), 0.007)


if __name__ == '__main__':
    unittest.main()
