# perceptron.py
import numpy as np


class PrimalPerceptron():
    """Implementation of the Primal Perceptron algorithm
    with margins.

    dim: int, dimension count of input data
    """

    def __init__(self, dim):
        """Creates weight vector w to match
        the input dimensionality.
        """
        self.w = [0.0 for val in range(dim)]

    def __str__(self):
        """Return current list of weights for printing."""
        return str(self.w)

    def classify(self, in_vals):
        """Computes signed dot product of input
        values with current weights.

        in_vals: list/array of input values
        """
        val = np.dot(self.w, in_vals)

        # Return signed value
        if val < 0:
            return -1
        else:
            return 1

    def update(self, in_vals, y):
        """Updates weight vector w according to
        w_k <- w_k + (y * x_k)

        in_vals: list/array of input values
        y: output label from training example
        """
        for k in range(len(in_vals)):
            self.w[k] += (y * in_vals[k])
