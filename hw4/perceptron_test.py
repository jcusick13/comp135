# perceptron_test.py
import unittest
from perceptron import *


class TestPerceptronFns(unittest.TestCase):

    #
    # PrimalPerceptron.calc_margin()
    #
    def test_margin_calculation(self):
        """Ensures proper margin calculation."""
        train_data = [[2, 4, 1], [3, 2, -1]]
        p = PrimalPerceptron(2)
        margin = p.calc_margin(train_data)

        self.assertEqual(round(margin, 4), 0.4039)

    def test_set_margin(self):
        """Ensures margin attribute is updated."""
        train_data = [[2, 4, 1], [3, 2, -1]]
        p = PrimalPerceptron(2)
        p.calc_margin(train_data)

        self.assertEqual(round(p.margin, 4), 0.4039)

    #
    # PrimalPerceptron.classify()
    #
    def test_output_1(self):
        """Ensures output label set to 1."""
        train_data = [2, 4]
        p = PrimalPerceptron(2)

        o, value = p.classify(train_data)
        self.assertEqual(o, 1)

    def test_output_0(self):
        """Ensures output label set to 0."""
        train_data = [1, 3]
        p = PrimalPerceptron(2)
        p.w = [-1, -1]

        o, value = p.classify(train_data)
        self.assertEqual(o, -1)

    #
    # PrimalPerceptron.update()
    #
    def test_weight_update(self):
        """Ensures weights are correctly updated after
        training example.
        """
        p = PrimalPerceptron(2)
        vals = [1, 2]
        y = 1

        p.update(vals, y)
        self.assertEqual(p.w, [1, 2])


if __name__ == '__main__':
    unittest.main()
