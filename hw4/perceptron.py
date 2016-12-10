# perceptron.py
import numpy as np
import math


class PrimalPerceptron():
    """Implementation of the Primal Perceptron algorithm
    with margins.
    """

    def __init__(self, dim):
        """Creates weight vector w to match
        the input dimensionality.

        dim: int, dimension count of input data
        """
        self.w = [0.0 for val in range(dim)]
        self.margin = 0.0

    def __str__(self):
        """Return current list of weights for printing."""
        return str(self.w)

    def calc_margin(self, train_data):
        """Calculates seperability margin as 0.1 * the
        average norm of training examples. Adds a constant
        feature 1 to all dataset example to account for the
        threshold.

        train_data: 2d-list, input training data
        """
        count = 0.0  # Total training examples
        running_tally = 0.0  # Sum of all norms

        # Sum norm over all examples
        for ex in train_data:

            # Calculate norm for individual example
            ex_tally = 0.0
            for ft in ex[:-1]:  # Ignore example label
                ex_tally += (ft ** 2)

            # Add constant value 1 ** 2 to each example
            ex_tally += 1.0

            # Root of sum of squares
            norm = math.sqrt(ex_tally)

            running_tally += norm
            count += 1.0

        margin = 0.1 * (running_tally / count)
        self.margin = margin
        return margin

    def classify(self, in_vals):
        """Computes signed dot product of input
        values with current weights.

        in_vals: list/array of input values
        """
        val = np.dot(self.w, in_vals)

        # Return signed value
        if val < 0:
            return -1, val
        else:
            return 1, val

    def update(self, in_vals, y):
        """Updates weight vector w according to
        w_k <- w_k + (y * x_k)

        in_vals: list/array of input values
        y: output label from training example
        """
        for k in range(len(in_vals)):
            self.w[k] += (y * in_vals[k])

    def train(self, train_data, i):
        """Learns weight vector from training
        examples with i iterations.

        train_data: Arff class of training data
        i: int, number of training iterations
        """
        # Calculate margin
        self.calc_margin(train_data.data)

        # Train
        for iteration in range(i):
            for ex in train_data.data:
                # Convert tuple to list
                ex_list = [x for x in ex[:-1]]
                # Add constant 1 to example to account for threshold
                example = ex_list + [1]
                o, value = self.classify(example)

                # Update weights if prediction incorrect/within margin
                if (o * value) < self.margin:
                    self.update(example, o)

        # Return final weight vector as hypothesis
        return self.w

    def test(self, test_data):
        """Labels examples in test_data according to current
        weight vector w, returns test dataset accuracy.

        test_data: Arff class of testing data
        """
        correct = 0.0
        total = 0.0

        for ex in test_data.data:
            # Convert tuple to list
            ex_list = [x for x in ex[:-1]]
            # Add constant 1 to example to account for threshold
            example = ex_list + [1]
            o, val = self.classify(example)

            # Compare labels, record result
            if o == int(ex[-1]):
                correct += 1.0

            total += 1.0

        return correct / total


class KernelPerceptron():
    """Implementation of Kernel Perceptron algorithm
    with both polynomial and RBF kernels.
    """

    def __init__(self, ex_ct, poly=False, d=0, RBF=False, s=0.0):
        """Initializes all zero alpha vector for length
        of the input example count.

        ex_ct: int, count of training examples
        poly: bool, switch to indicate polynomial kernel
        d: int, exponential parameter of polynomial kernel
        RBF: bool, switch to indicated RBF kernel
        s: float, RBF kernel parameter
        """
        self.alpha = [0 for x in range(ex_ct)]
        self.poly = poly
        self.d = d
        self.rbf = RBF
        self.s = s
        self.margin = 0.0

        if not poly and not RBF:
            raise Exception('Polynomial or RBF kernel must be selected.')

    def __str__(self):
        """Returns alpha vector as a string for printing."""
        return str(self.alpha)

    def poly_kernel(self, u, v):
        """Computes polynomial kernel of two vectors u and v:
        (u * v + 1)^d

        u: list/array, input vector of numeric values
        v: list/array, input vector of numeric values (same length as u)
        """
        return (np.dot(u, v) + 1) ** self.d

    def rbf_kernel(self, u, v):
        """Computes the radial basis function (gaussian) kernel of
        two vectors u and v:
        e ^ (- (||u - v||^2) / (2s^2))

        u: list/array, input vector of numeric values
        v: list/array, input vector of numeric values (same length as u)
        """
        # Squared norm difference
        diff = np.subtract(u, v)
        top = np.sum(np.square(diff))

        # Sigma parameter
        bottom = 2.0 * (self.s ** 2.0)

        return math.exp(- (top / bottom))

    def calc_margin(self, train_data):
        """Calculates the seperability margin as
        0.1 * average kernel value. Uses either polynomial
        or RBF kernel as specified by class parameter.

        train_data: 2-d list, input training data
        """
        count = 0.0  # Total training examples
        running_tally = 0.0  # Sum of all sqrt(kernel)

        # Sum sqrt(kernel) over all examples
        for ex in train_data:
            # Calculate kernel, add to running tally
            if self.poly:
                kernel = self.poly_kernel(ex[:-1], ex[:-1])
            else:
                kernel = self.rbf_kernel(ex[:-1], ex[:-1])

            running_tally += math.sqrt(kernel)
            count += 1.0

        margin = 0.1 * (running_tally / count)
        self.margin = margin
        return margin

    def classify(self, in_vals, train):
        """Computes signed classifer function of input values

        in_vals: list/array, training or test example to classify
        train: 2-d list/array, full training dataset
        """
        val = 0.0

        # Compare example to every training instance for classification
        for k in range(len(train)):
            u = train[k][:-1]  # k-th dataset example
            v = in_vals[:-1]  # example to classify

            # (alpha_k * y_k * kernel value)
            if self.poly:
                val += self.alpha[k] * int(train[k][-1]) * self.poly_kernel(u, v)
            else:
                val += self.alpha[k] * int(train[k][-1]) * self.rbf_kernel(u, v)

        # Return signed value
        if val < 0:
            return -1, val
        else:
            return 1, val

    def update(self, i):
        """Updates alpha vector with
        a_i <- a_i + 1

        i: int, alpha vector element to be updated
        """
        self.alpha[i] += 1

    def train(self, traindata, i):
        """Learns alpha vector values from input dataset.

        traindata: Arff class, training dataset
        i: int, number of training iterations
        """
        # Calculate margin
        self.calc_margin(traindata.data)

        # Train
        for iteration in range(i):
            curr_example = 0
            for ex in traindata.data:
                o, val = self.classify(ex, traindata.data)

                # Update alpha if prediction incorrect/within margin
                if (o * val) < self.margin:
                    self.update(curr_example)

                # Update current example count
                curr_example += 1

        # Return final alpha vector as hypothesis
        return self.alpha

    def test(self, traindata, testdata):
        """Labels examples in test dataset according to
        alpha weight vector.

        traindata: Arff class, training dataset
        testdata: Arff class, testing dataset
        """
        correct = 0.0
        total = 0.0

        for example in testdata.data:
            o, val = self.classify(example, traindata.data)

            # Compare labels, record result
            if o == int(example[-1]):
                correct += 1.0
            total += 1.0

        return correct / total
