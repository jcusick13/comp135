# knn_test.py
import unittest
from arff_reader import *
from knn import *


class TestKnnMethods(unittest.TestCase):

    #
    # entropy()
    #

    def test_entropy_zero_non_neg(self):
        """Output does not have a negative sign when 0."""
        collection = [100, 0]

        self.assertEqual(entropy(collection), 0)

    def test_entropy_one(self):
        """Output is 1.0 when two inputs are the same."""
        collection = [15, 15]

        self.assertEqual(entropy(collection), 1.0)

    #
    # euclidean_dist()
    #

    def test_euclid_dist_0_pos_input(self):
        """Output is correct with all positive input. """
        x = (1, 2, 3, 4, 5)
        y = (1, 2, 3, 4, 5)
        weights = [1, 1, 1, 1, 1]

        self.assertEqual(euclidean_dist(x, y, weights), 0)

    def test_euclid_dist_5_neg_input(self):
        """Output is correct with all negative input. """
        x = (-2, -2)
        y = (-6, -5)
        weights = [1, 1]

        self.assertEqual(euclidean_dist(x, y, weights), 5)

    def test_euclid_dist_0_weight(self):
        """Distance is 0 since weights disregard far attribute."""
        x = (5, -99)
        y = (5, 100)
        weights = [1, 0]

        self.assertEqual(euclidean_dist(x, y, weights), 0)

    #
    # find_neighbors()
    #

    def test_find_neighbors_tie_keep_one(self):
        """Only keep first neighbor seen during tie in
        Euclidean distance.
        """
        train = [(1, 1, 'x'), (2, 4, 'x'), (2, 4, 'y')]
        inst = (1, 2, 'z')
        weights = [1, 1]

        self.assertEqual(find_neighbors(train, inst, 2, weights),
                         [(1, 1, 'x'), (2, 4, 'x')])

    def test_find_neighbors_tie_keep_all(self):
        """Keep all neighbors involved during tie in
        Euclidean distance.
        """
        train = [(5, 2, 'a'), (-1, 3, 'b'), (-1, 3, 'c'), (1, 1, 'd')]
        inst = (-2, 4, 'z')
        weights = [1, 1]

        self.assertEqual(find_neighbors(train, inst, 2, weights),
                         [(-1, 3, 'b'), (-1, 3, 'c')])

    def test_find_neighbors_ensure_cleaned(self):
        """Remove distance score that was appended during
        function for final neighbors being returned.
        """
        train = [(1, 2, 3, 'a'), (4, 5, 6, 'b')]
        inst = (1, 5, 3, 'c')
        weights = [1, 1, 1]

        self.assertEqual(len(find_neighbors(train, inst, 1, weights)[0]), 4)

    def test_find_neighbors_use_weights(self):
        """Weights of 0 are applied so the only neighbor to be
        compared to is the farthest possible neighbor, which now
        returns as a match.
        """
        train = [(0, 0, 10, 'a'), (1, 1, 20, 'a'), (25, 25, 1, 'b')]
        inst = (0, 0, 0, 'a')
        weights = [0, 0, 1]

        self.assertEqual(find_neighbors(train, inst, 1, weights),
                         [(25, 25, 1, 'b')])

    #
    # gain()
    #

    def test_gain(self):
        """Information gain outputs correct example on set of inputs."""
        collection = [9, 5]
        attr = [[2, 3], [4, 0], [3, 2]]

        self.assertEqual(gain(collection, attr), 0.2467498197744391)

    #
    # knn_work()
    #

    def test_knn_work_equivalent(self):
        """Output a score of 1.0 for completely correctly
        classified dataset.
        """
        train = [(1, 1, 'a'), (2, 2, 'a'), (1.5, 1.5, 'a'),
                 (5, 5, 'b'), (6, 7, 'b'), (7, 5, 'b')]
        test = [(0, 0, 'a'), (10, 10, 'b')]
        field_names = ['a01', 'a02', 'c']

        self.assertEqual(knn_work(train, test, 3, field_names, None, None),
                         1.0)

    def test_knn_work_05(self):
        """Output a score of 0.5 when only half of data is
        correctly classified.
        """
        train = [(1, 1, 'a'), (2, 2, 'a'), (1.5, 1.5, 'a'),
                 (5, 5, 'b'), (6, 7, 'b'), (7, 5, 'b')]
        test = [(0, 0, 'a'), (10, 10, 'a')]
        field_names = ['a01', 'a02', 'c']

        self.assertEqual(knn_work(train, test, 2, field_names, None, None),
                         0.5)

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


if __name__ == '__main__':
    unittest.main()
