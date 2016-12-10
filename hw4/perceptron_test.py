# perceptron_test.py
import unittest
from perceptron import *


class TestPerceptronFns(unittest.TestCase):

    #
    # PrimalPerceptron.calc_margin()
    #
    def test_primal_margin_calculation(self):
        """Ensures proper margin calculation."""
        train_data = [[2, 4, 1], [3, 2, -1]]
        p = PrimalPerceptron(2)
        margin = p.calc_margin(train_data)

        self.assertEqual(round(margin, 4), 0.4162)

    def test_primal_set_margin(self):
        """Ensures margin attribute is updated."""
        train_data = [[2, 4, 1], [3, 2, -1]]
        p = PrimalPerceptron(2)
        p.calc_margin(train_data)

        self.assertEqual(round(p.margin, 4), 0.4162)

    #
    # PrimalPerceptron.classify()
    #
    def test_primal_output_1(self):
        """Ensures output label set to 1."""
        train_data = [2, 4]
        p = PrimalPerceptron(2)

        o, value = p.classify(train_data)
        self.assertEqual(o, 1)

    def test_primal_output_0(self):
        """Ensures output label set to 0."""
        train_data = [1, 3]
        p = PrimalPerceptron(2)
        p.w = [-1, -1]

        o, value = p.classify(train_data)
        self.assertEqual(o, -1)

    #
    # PrimalPerceptron.update()
    #
    def test_primal_weight_update(self):
        """Ensures weights are correctly updated after
        training example.
        """
        p = PrimalPerceptron(2)
        vals = [1, 2]
        y = 1

        p.update(vals, y)
        self.assertEqual(p.w, [1, 2])

    #
    # KernelPerceptron.poly_kernel()
    #
    def test_poly_kernel_linear_calc(self):
        """Ensure correct output for kernel calculation."""
        u = [3, 2, 1]
        v = [1, 2, 3]

        kp = KernelPerceptron(2, poly=True, d=1)
        kernel = kp.poly_kernel(u, v)

        self.assertEqual(kernel, 11)

    def test_poly_kernel_power_calc(self):
        """Ensure correctness when raised to a power."""
        u = [3, 2, 1]
        v = [1, 2, 3]

        kp = KernelPerceptron(2, poly=True, d=3)
        kernel = kp.poly_kernel(u, v)

        self.assertEqual(kernel, 1331)

    def test_rbf_kernel_sigma_2(self):
        """Ensure correctness when sigma = 2."""
        u = [3, 2, 1]
        v = [1, 2, 3]

        kp = KernelPerceptron(2, RBF=True, s=2.0)
        kernel = kp.rbf_kernel(u, v)

        self.assertEqual(round(kernel, 4), 0.3679)

    #
    # KernelPerceptron.calc_margin()
    #
    def test_poly_kernel_margin(self):
        """Ensure correct margin calculation with poly kernel."""
        train_data = [[2, 4, 6, -1], [1, 2, 3, 1]]
        kp = KernelPerceptron(2, poly=True, d=1)
        margin = kp.calc_margin(train_data)

        self.assertEqual(round(margin, 4), 0.5711)

    def test_poly_kernel_margin_update(self):
        """Checks that margin is updated after calculation."""
        train_data = [[2, 4, 6, -1], [1, 2, 3, 1]]
        kp = KernelPerceptron(2, poly=True, d=1)
        kp.calc_margin(train_data)

        self.assertEqual(round(kp.margin, 4), 0.5711)

    def test_rbf_kernel_margin(self):
        """Ensure correct margin calculation with rbf kernel."""
        train_data = [[3, 2, 1, 1], [1, 2, 3, 1]]
        kp = KernelPerceptron(2, RBF=True, s=1.0)
        margin = kp.calc_margin(train_data)

        self.assertEqual(round(margin, 1), 0.1)

    #
    # KernelPerceptron.classify()
    #
    def test_poly_kernel_classify_initial(self):
        """Label should be 1 because of zero
        initialized alpha vector."""
        train_data = [[2, 4, 6, -1], [1, 2, 3, 1]]
        example = [2, 4, 6, -1]
        kp = KernelPerceptron(2, poly=True, d=1)
        o, val = kp.classify(example, train_data)

        self.assertEqual(o, 1)


if __name__ == '__main__':
    unittest.main()
