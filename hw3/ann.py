# ann.py
import math
import numpy as np


class Node():
    """Individual node for use within a neural network.
    Can be used anywhere in net (i.e. input, output, hidden).
    """

    def __init__(self, value=0.0, inputs=0):
        """All nodes default to value of zero and random weight
        uniformly sampled from range [-0.1, 0.1].

        value: float, initial value of the node.

        inputs: int, number of nodes from the previous layer that
                have a connection to this node.
        """
        self.value = value
        if inputs == 0:
            self.weights = None
        else:
            self.weights = np.random.uniform(-0.1, 0.1, inputs)

    def update_value(self, prev_values):
        """Sets the the output value (x_i) of a node, using the
        sigmoid function.

        prev_values: list, final value (x_i) of each node
                    in the previous layer.
        """
        s = self._s_i(prev_values)
        self.value = self._sigmoid(s)

    def _s_i(self, prev_values):
        """Calculates s_i, the sum of all values (multiplied by
        their weights) of connected nodes from the previous layer.

        prev_values: list, final value (x_i) of each node
                    in the previous layer.
        """
        return np.dot(self.weights, prev_values)

    def _sigmoid(self, i):
        """Calculates the sigmoid of i (x_i), returns as float.

        i: float/int
        """

        # Ensure float input
        i = i / 1.0
        return 1.0 / (1.0 + math.exp(-i))

    def _sigmoid_p(self, i):
        """Calculates the derivative of the sigmoid
        of i, returns as float.

        i: float/int
        """

        # Ensure float input
        i = i / 1.0
        return i * (1.0 - i)


class Layer():
    """Collection of nodes within one layer of a neural network."""

    def __init__(self, nodes, prev=None, follow=None):
        """Initializes array of individual nodes.

        nodes: int, the number of nodes to be contained in the
                layer (i.e. layer width)

        prev: Layer, the layer immediately preceeding (closer to
              input nodes) the current layer.

        follow: Layer, the layer immediately after (closer to
                output nodes) the current layer.
        """
        self.prev = prev
        self.follow = follow
        self.nodes = np.array([Node() for x in nodes], dtype=object)

