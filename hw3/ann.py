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

    def __str__(self):
        return str(self.value)

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

    def __init__(self, nodes, value=None, prev_inputs=None):
        """Initializes array of individual nodes.

        nodes: int, the number of nodes to be contained in the
                layer (i.e. layer width)
        value: list, set of float values to initialize nodes to
        prev_inputs: int, number of nodes in the previous layer
        """
        if not value:
            # Initialize nodes to value of 0.0
            self.nodes = [Node(inputs=prev_inputs) for x in range(nodes)]
        else:
            # Initialize nodes to value list passed in
            assert(nodes == len(value))
            self.nodes = [Node(value=value[x], inputs=prev_inputs)
                          for x in range(nodes)]

    def __str__(self):
        """Prints a list of node values for each node in the layer."""
        return str([i.value for i in self.nodes])

    def values(self):
        """Returns list of node output values for layer."""
        return [node.value for node in self.nodes]


class NeuralNet():
    """Neural network implementation using sigmoid activation function
    with backpropagation used to determine error and weight updates.
    """

    def __init__(self, inputs, outputs, width, depth):
        """Initializes network structure. Every node between two adjacent
        layers are directly connected to each other.

        inputs: int, number of input nodes
        outputs: int, number of output nodes
        width: int, number of nodes per hidden layer
        depth: int, number of hidden layers between input and output layers
        """

        self.inputs = inputs
        self.outputs = outputs
        self.w = width
        self.d = depth
        self.layers = []

        # Add first hidden layer (unsure of previous layer node count) -
        #   Just a placeholder to be replaced once input dimensionality is
        #   determined and correct amount of weight values can be created
        self.layers.append(Layer(self.w))

        # Add hidden layers 2 to max (previous layer node count is known)
        hidden = Layer(self.w, prev_inputs=self.w)
        [self.layers.append(hidden) for x in range(self.d - 1)]

        # Add output layer (previous layer node count is known)
        self.layers.append(Layer(self.outputs, prev_inputs=self.w))

    def __str__(self):
        return str([l.__str__() for l in self.layers])

    def propagate_forward(self, in_values):
        """Creates input layer to network then walks forward
        from input to output layer calculating the output value
        of each node along the way.

        in_values: list, initialization values for input nodes
        """

        # Insert input nodes as first layer in network
        input_layer = Layer(len(in_values), value=in_values)
        self.layers.insert(0, input_layer)

        # Recreate first hidden layer (to generate correct num. of weights)
        self.layers[1] = Layer(self.w, prev_inputs=len(in_values))

        # Create list of node values from previous layer
        prev_values = in_values

        # Walk forward through layers, update values (skip input layer)
        for layer in self.layers[1:]:
            print prev_values
            for node in layer.nodes:
                node.update_value(prev_values)
            prev_values = layer.values()
