# knn_test.py
import unittest
from knn import *


class TestKnnMethods(unittest.TestCase):

    #
    # euclidean_dist()
    #

    def test_euclid_dist_0_pos_input(self):
        """Output is correct with all positive input. """
        x = (1, 2, 3, 4, 5)
        y = (1, 2, 3, 4, 5)

        self.assertEqual(euclidean_dist(x, y), 0)

    def test_euclid_dist_5_neg_input(self):
        """Output is correct with all negative input. """
        x = (-2, -2)
        y = (-6, -5)

        self.assertEqual(euclidean_dist(x, y), 5)

    #
    # find_neighbors()
    #

    def test_find_neighbors_tie_keep_one(self):
        """Only keep first neighbor seen during tie in
        Euclidean distance.
        """
        train = [(1, 1, 'x'), (2, 4, 'x'), (2, 4, 'y')]
        inst = (1, 2, 'z')

        self.assertEqual(find_neighbors(train, inst, 2),
                         [(1, 1, 'x'), (2, 4, 'x')])

    def test_find_neighbors_tie_keep_all(self):
        """Keep all neighbors involved during tie in
        Euclidean distance.
        """
        train = [(5, 2, 'a'), (-1, 3, 'b'), (-1, 3, 'c'), (1, 1, 'd')]
        inst = (-2, 4, 'z')

        self.assertEqual(find_neighbors(train, inst, 2),
                         [(-1, 3, 'b'), (-1, 3, 'c')])

    def test_find_neighbors_ensure_cleaned(self):
        """Remove distance score that was appended during
        function for final neighbors being returned.
        """
        train = [(1, 2, 3, 'a'), (4, 5, 6, 'b')]
        inst = (1, 5, 3, 'c')

        self.assertEqual(len(find_neighbors(train, inst, 1)[0]), 4)

    #
    # majority_count()
    #

    def test_majority_count_y(self):
        """Output correct majority classification of 'y'. """
        exp = [(1, 'y'), (2, 'n'), (3, 'y')]

        self.assertEqual(majority_count(exp), 'y')

    def test_majority_count_tie(self):
        """Output first classification seen in case of tie
        in count.
        """
        exp = [(1, 'y'), (2, 'n'), (3, 'y'), (4, 'n')]

        self.assertEqual(majority_count(exp), 'y')

    #
    # knn()
    #

    def test_knn_equivalent(self):
        """Output a score of 1.0 for completely correctly
        classified dataset.
        """
        train = [(1, 1, 'a'), (2, 2, 'a'), (1.5, 1.5, 'a'),
                 (5, 5, 'b'), (6, 7, 'b'), (7, 5, 'b')]
        test = [(0, 0, 'a'), (10, 10, 'b')]

        self.assertEqual(knn(train, test, 3), 1.0)

    def test_knn_05(self):
        """Output a score of 0.5 when only half of data is
        correctly classified.
        """
        train = [(1, 1, 'a'), (2, 2, 'a'), (1.5, 1.5, 'a'),
                 (5, 5, 'b'), (6, 7, 'b'), (7, 5, 'b')]
        test = [(0, 0, 'a'), (10, 10, 'a')]

        self.assertEqual(knn(train, test, 2), 0.5)


if __name__ == '__main__':
    unittest.main()
