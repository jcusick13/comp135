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

    #
    # Node._delta_n()
    #
    def test_delta_n_output(self):
        """Ensures correct output of delta value for nodes
        in the network output layer.
        """
        n = Node(value=0.823)
        self.assertEqual(round(n._delta_n(0), 3), 0.120)

    #
    # Node._delta_i()
    #
    def test_delta_i_output_multi(self):
        """Ensures correct output of delta value for nodes
        within a hidden layer in the network that have multiple
        nodes in the k-th layer.

        Node is also first node in its layer.
        """

        # Current node
        n = Node(value=0.993)

        # K-layer nodes
        k1 = Node(value=0.767)
        k1.delta = 0.021
        k1.weights = [0.6, 0.6]

        k2 = Node(value=0.767)
        k2.delta = 0.021
        k2.weights = [0.6, 0.6]

        # K-layer
        k = Layer(0)
        k.nodes.append(k1)
        k.nodes.append(k2)

        self.assertEqual(round(n._delta_i(k, 0), 6), 0.000175)

    def test_delta_i_output_single(self):
        """Ensures correct output of the delta value for nodes
        within a hidden layer in the network that have a single
        node in the k-th layer.

        Node is last node in its layer.
        """

        # Current node
        n = Node(value=0.767)

        # K-layer node
        k1 = Node(value=0.823)
        k1.delta = 0.120
        k1.weights = [0.2, 1.0]

        # K-layer
        k = Layer(0)
        k.nodes.append(k1)

        self.assertEqual(round(n._delta_i(k, 1), 3), 0.021)

    #
    # Node.update_weight()
    #
    def test_node_update_weight(self):
        """Ensures correct output during weight update process."""
        n = Node(value=0.823)
        n.weights = [1.0, 1.0]
        n.delta = 0.120
        eta = 0.1
        x_j = 0.767

        n.update_weight(eta, x_j, 0)
        weight = n.weights[0]
        self.assertEqual(round(weight, 4), 0.9908)

    #
    # NeuralNet.propagate_forward()
    #
    def test_propagate_forward(self):
        """Ensures correct final network output of forward
        propagation of x_i values.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 1, 2, 2)

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        for node in nn.layers[3].nodes:
            node.weights = [1.0, 1.0]

        nn.propagate_forward([2, 3], test=True)
        model_output = nn.layers[-1].nodes[0].value

        self.assertEqual(round(model_output, 3), 0.823)

    def test_propagate_forward_no_hidden(self):
        """Ensures correct output when the network
        consists of a single input and output layer
        i.e. - no hidden layers.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 1, 0, 0)

        for node in nn.layers[-1].nodes:
            node.weights = [1.0, 1.0]

        nn.propagate_forward([2, 3], test=True)
        model_output = nn.layers[-1].nodes[0].value

        self.assertEqual(round(model_output, 3), 0.993)

    #
    # NeuralNet.propagate_backward()
    #
    def test_propagate_backward_last_hidden(self):
        """Ensures correct weight updates are made to the final
        (closest to output layer) hidden layer.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 1, 2, 2)
        nn.eta = 0.1

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        for node in nn.layers[3].nodes:
            node.weights = [1.0, 1.0]

        # Walk forward
        nn.propagate_forward([2, 3], test=True)

        # Walk backward
        nn.propagate_backward([0])

        test_weight = nn.layers[-1].nodes[0].weights[0]
        self.assertEqual(round(test_weight, 4), 0.9901)

    def test_propagate_backward_first_hidden(self):
        """Ensures correct weight updates are made to the first
        (closest to input layer) hidden layer.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 1, 2, 2)
        nn.eta = 0.1

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        for node in nn.layers[3].nodes:
            node.weights = [1.0, 1.0]

        # Walk forward
        nn.propagate_forward([2, 3], test=True)

        # Walk backward
        nn.propagate_backward([0])

        test_weight = nn.layers[1].nodes[0].weights[0]
        self.assertEqual(round(test_weight, 6), 0.999983)

    #
    # NeuralNet.update_weights()
    #
    def test_net_weight_update(self):
        """Ensures correct weight to output
        node after walking through the network forwards
        and backwards once.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 1, 2, 2)
        nn.eta = 0.1

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        for node in nn.layers[3].nodes:
            node.weights = [1.0, 1.0]

        nn.update_weights([2, 3], [0], test=True)

        test_weight = nn.layers[-1].nodes[0].weights[0]
        self.assertEqual(round(test_weight, 4), 0.9901)

    #
    # NeuralNet.assign_output()
    #
    def test_find_highest_value_node_last(self):
        """Ensures that the last output node is correctly
        selected for prediction.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 2, 2, 2)
        nn.eta = 0.1

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        nn.layers[3].nodes[0].weights = [0.0, 0.0]
        nn.layers[3].nodes[1].weights = [1.0, 1.0]

        val = nn.assign_output([2, 3], test=True)
        self.assertEqual(val, '01')

    def test_find_highest_value_node_first(self):
        """Ensures that the first output node is correctly
        selected for prediction.
        """
        nn = NeuralNet(0, 0, '', '', blank=True)
        nn.create_net(2, 2, 2, 2)
        nn.eta = 0.1

        # Override weights to static value for reproducibility
        for node in nn.layers[2].nodes:
            node.weights = [0.6, 0.6]

        nn.layers[3].nodes[0].weights = [1.0, 1.0]
        nn.layers[3].nodes[1].weights = [0.0, 0.0]

        val = nn.assign_output([2, 3], test=True)
        self.assertEqual(val, '10')


if __name__ == '__main__':
    unittest.main()
