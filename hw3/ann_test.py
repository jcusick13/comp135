# ann_test.py
import unittest
from ann import *


class TestAnnFunctions(unittest.TestCase):

    #
    # Node.__init__()
    #

    def test_node_weight_init(self):
        """Ensures weights are only initialized for a node
        when passed in an input value.
        """
        n = Node(value=2.0)
        self.assertEqual(n.weights, None)

    def test_node_weight_range_min(self):
        """Ensures random weight values are initialized
        within the correct range.
        """
        n = Node(inputs=6)
        for i in n.weights:
            self.assertGreaterEqual(i, -0.1)

    def test_node_weight_range_max(self):
        """Ensures random weight values are initialized
        within the correct range.
        """
        n = Node(inputs=3)
        for i in n.weights:
            self.assertLess(i, 0.1)

    #
    # Node.update_value()
    #

    def test_update_value(self):
        """Ensures node output value is correctly calculated."""
        n = Node(inputs=2)
        # Override weights to static value for reproducibility
        n.weights = [1, 1]
        n.update_value([2, 3])

        self.assertEqual(round(n.value, 3), 0.993)

    #
    # Node._sigmoid()
    #

    def test_sigmoid_convert_int(self):
        """Ensures output is the same regardless if input
        is given as int or float.
        """
        n = Node()
        self.assertEqual(n._sigmoid(5), n._sigmoid(5.0))

    #
    # Node._sigmoid_p()
    #

    def test_sigmoid_p_output(self):
        """Ensures correct output of sigmoid derivative."""
        n = Node()
        self.assertEqual(round(n._sigmoid_p(0.993), 3), 0.007)


if __name__ == '__main__':
    unittest.main()
