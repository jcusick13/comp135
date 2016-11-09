# ann.py
import math


def sigmoid(i):
    """Calculates the sigmoid of i, returns as float.

    i: float/int
    """

    # Ensure float input
    i = i / 1.0

    return 1.0 / (1.0 + math.exp(-i))


def sigmoid_p(i):
    """Calculates the derivative of the sigmoid
    of i, returns as float.

    i: float/int
    """

    # Ensure float input
    i = i / 1.0

    return i * (1.0 - i)
