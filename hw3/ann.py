# ann.py
import math
import numpy as np
from arff_reader import Arff


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

        # Delta value for backproagation (placeholder during init)
        self.delta = -999.9

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

    def _delta_n(self, y):
        """Calculates the delta value used during backpropagation
        of nodes in the output layer.

        y: float/int, target function value for x_N
        """
        sig_p = self._sigmoid_p(self.value)

        self.delta = (-sig_p) * (y - self.value)
        return self.delta

    def _delta_i(self, k, index):
        """Calculates the delta value used during backpropagation
        for nodes in hidden layers.

        k: Layer, the layer immediately after (closer to the output layer)
            the layer in which this node resides.
        index: int, zero-based index of this node's location within
            its layer
        """
        sig_p = self._sigmoid_p(self.value)
        sum_k = 0.0

        # Gather values from the k-layer
        for node in k.nodes:
            sum_k += (node.delta * node.weights[index])

        self.delta = sig_p * sum_k
        return self.delta

    def update_weight(self, eta, x_j, index):
        """Updates the node's weight during backpropagation using
        gradient descent.

        eta: float, learning rate
        x_j: float, sigmoid output of the x_j node
        index: int, zero-based index of this node's weight under
            consideration
        """
        w = self.weights[index] - (eta * self.delta * x_j)
        self.weights[index] = w


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
    with backpropagation used for determining error and weight updates.
    """

    def __init__(self, width, depth, train, test, eta=0.1, blank=False):
        """Determines network parameters. Hidden layer width and depth
        are user-defined while input dimension count and output label
        count are taken from the training file. Train and test datasets
        are expected to have the same feature/output count.

        width: int, number of nodes per hidden layer
        depth: int, number of hidden layers between input and output layers
        train: arff file, training data set with numeric only attributes
        test: arff file, test data set with numeric only attributes
        eta: float, learning rate
        blank: bool, switch used to initialize empty net, used
                for unit testing
        """
        if not blank:
            self.w = width
            self.d = depth
            self.train = Arff(train)
            self.test = Arff(test)

            # Input dimension count (subtract output colmn name)
            self.inputs = len(self.train.field_names) - 1

            # Output label count
            self.outputs = len(self.train.classes.keys())

            # Initialize network
            self.create_net(self.inputs, self.outputs, self.w, self.d)

    def __str__(self):
        return str([l.__str__() for l in self.layers])

    def create_net(self, inputs, outputs, width, depth):
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
        if self.d > 0:
            self.layers.append(Layer(self.w))

            # Add hidden layers 2 to max (previous layer node count is known)
            hidden = Layer(self.w, prev_inputs=self.w)
            [self.layers.append(hidden) for x in range(self.d - 1)]

            # Add output layer (previous layer node count is known)
            self.layers.append(Layer(self.outputs, prev_inputs=self.w))

        # No hidden layers, input connected directly to output
        else:
            self.layers.append(Layer(self.outputs, prev_inputs=self.inputs))

    def propagate_forward(self, in_values, test=False):
        """Creates input layer to network then walks forward
        from input to output layer calculating the output value
        of each node along the way.

        in_values: list, initialization values for input nodes
        test: bool, switch to help set weights during testing
        """

        # Insert input nodes as first layer in network
        input_layer = Layer(len(in_values), value=in_values)
        self.layers.insert(0, input_layer)

        # Recreate first hidden layer (to generate correct weight values)
        if self.d > 0:
            self.layers[1] = Layer(self.w, prev_inputs=len(in_values))

        if test:
            # Reset weights for 1st hidden layer to 1 for testing
            for node in self.layers[1].nodes:
                node.weights = [1.0 for i in range(len(in_values))]

        # Create list of node values from previous layer
        prev_values = in_values

        # Walk forward through layers, update node values (skip input layer)
        for layer in self.layers[1:]:
            for node in layer.nodes:
                node.update_value(prev_values)
            prev_values = layer.values()

    def propagate_backward(self, outputs):
        """Walks backward through the network, updating the node weights in
        each layer using a stochastic gradient decent update function.

        eta: float, learning rate
        outputs: list, float values of target function outputs
        """
        # Calculate delta value for output layer
        i = 0
        for node in self.layers[-1].nodes:
            node._delta_n(outputs[i])
            i += 1

        # Walk backwards
        lyr = -2
        for layer in reversed(self.layers[1:]):  # Skip input layer;
                                                # nodes contain no weights

            # Calculate delta values for nodes in preceding (closer to
            #   inputs) layer
            node_index = 0
            for node in self.layers[lyr].nodes:
                node._delta_i(layer, node_index)
                node_index += 1

            # Node connections that require weight updates
            weight_width = len(self.layers[lyr].nodes)

            # Gradient descent update for nodes in current layer
            for node in layer.nodes:
                for j in range(weight_width):
                    node.update_weight(self.eta, node.value, j)

            # Update layer index
            lyr -= 1

    def update_weights(self, in_values, outputs, test=False):
        """Updates network weights given a training input example

        in_values: list, initialization value for input nodes
        eta: float, learning rate,
        outputs: list, float values of target function outputs
        test: bool, switch to help set weights during testing
        """
        self.propagate_forward(in_values, test)
        self.propagate_backward(outputs)

    def assign_output(self, in_values, test=False):
        """Calculates value at each node in network, returning
        a one-hot encoded output value.

        in_values: list, initialization value for input nodes
        test: bool, switch to help set weights during testing
        """
        # Walk forward through net
        self.propagate_forward(in_values, test)

        # Find node index with highest value
        max_value = max(self.layers[-1].values())
        max_index = self.layers[-1].values().index(max_value)

        # Encode output
        onehot = ''
        for i in range(self.outputs):
            if i == max_index:
                onehot += '1'
            else:
                onehot += '0'

        return onehot
