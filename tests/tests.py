import cvxpy as cp
import numpy as np
import unittest

from verifiNN.models.network import Network
from verifiNN.verifier import LPVerifier


class TestNetwork(unittest.TestCase):

    def test_init(self):
        W1 = np.array([[1, 2, 3], [-4, -5, -6]])
        b1 = np.array([5, 6])
        W2 = np.array([[7, 8], [9, 10]])
        b2 = np.array([11, 12])

        weights = [W1, W2]
        biases = [b1, b2]

        nw = Network(weights, biases, activation='Id')
        self.assertTrue((nw.weights[0] == W1).all())
        self.assertTrue((nw.weights[1] == W2).all())
        self.assertTrue((nw.biases[0] == b1).all())
        self.assertTrue((nw.biases[1] == b2).all())

    def test_get_output_Id(self):
        W1 = np.array([[1, 2, -1], [3, 4, -3]])
        b1 = np.array([5, 6])
        W2 = np.array([[7, 8], [9, 10]])
        b2 = np.array([11, 12])

        weights = [W1, W2]
        biases = [b1, b2]
        network = Network(weights, biases, activation='Id')

        x = np.array([1, 1, 1])
        self.assertEqual(network.get_output(x)[0], 140)
        self.assertEqual(network.get_output(x)[1], 175)

    def test_get_output_ReLU(self):
        W1 = np.array([[1, 2, 3], [-4, -5, -6]])
        b1 = np.array([5, 6])
        W2 = np.array([[7, 8], [-9, -10]])
        b2 = np.array([11, 12])

        weights = [W1, W2]
        biases = [b1, b2]
        network = Network(weights, biases, activation='ReLU')

        x = np.array([1, 1, 1])
        self.assertEqual(network.get_output(x)[0], 88)
        self.assertEqual(network.get_output(x)[1], 0)

    def test_classify(self):
        W1 = np.array([[1, 2, -1], [3, 4, -3]])
        b1 = np.array([5, 6])
        W2 = np.array([[7, 8], [9, 10]])
        b2 = np.array([11, 12])

        weights = [W1, W2]
        biases = [b1, b2]
        network = Network(weights, biases, activation='Id')

        x = np.array([1, 1, 1])
        label = network.classify(x)
        self.assertEqual(label, 1)


class TestLPVerifierInFeasible(unittest.TestCase):

    W1 = np.array([[0.995, -0.100], [0.100, 0.995]])
    b1 = np.array([-1, 0])
    W2 = np.array([[0.622, 0.783], [-0.783, 0.622]])
    b2 = np.array([-2, 0])

    weights = [W1, W2]
    biases = [b1, b2]
    network = Network(weights, biases, activation='ReLU')

    epsilon = 0.5
    x_0 = np.array([1, 1])

    def test_get_activation_patters(self):
        vf = LPVerifier()
        vf.network = self.network
        patterns = vf.get_activation_pattern(self.x_0)
        for p in patterns:
            self.assertTrue((p == [0, 1]).all())

    def test_generate_decision_variables(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()
        self.assertEqual(vf.variables['z_0'].shape[0], 2)
        self.assertEqual(vf.variables['z_1'].shape[0], 2)
        self.assertEqual(vf.variables['z_2'].shape[0], 2)

        self.assertEqual(vf.variables['z_hat_1'].shape[0], 2)
        self.assertEqual(vf.variables['z_hat_2'].shape[0], 2)

    def test_generate_region_constraints(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()
        reg_cons = vf.generate_region_constraints(self.x_0, self.epsilon)
        self.assertEqual(len(reg_cons), 2)

    def test_generate_affine_constraints(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()
        aff_cons = vf.generate_affine_constraints()
        self.assertEqual(len(aff_cons), 2)

    def test_generate_ReLU_constraints(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()
        ReLU_cons = vf.generate_ReLU_constraints(self.x_0)
        self.assertEqual(len(ReLU_cons), 4)

    def test_generate_safety_set_constraints(self):
        l0 = self.network.classify(self.x_0)
        self.assertEqual(l0, 1)

        label = 0
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()
        ss_cons = vf.generate_safety_set_constraint(l0, label)
        self.assertIsInstance(ss_cons, cp.constraints.nonpos.Inequality)

    def test_solve_infeasible(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()

        obj = cp.Minimize(1)
        result = vf._solve(self.x_0, self.epsilon, obj)
        self.assertEqual(result['problem_status'], 'infeasible')
        self.assertEqual(result['optimal_value'], np.inf)
        self.assertIsNone(result['adversarial_example'])

    def test_verify_epsilon_robustness_infeasible(self):
        vf = LPVerifier()
        result = vf.verify_epsilon_robustness(
            self.network, self.x_0, self.epsilon)
        self.assertEqual(result['problem_status'], 'infeasible')
        self.assertEqual(result['optimal_value'], np.inf)
        self.assertIsNone(result['adversarial_example'])

        self.assertEqual(result['verification_status'], 'unverified')
        self.assertEqual(result['robustness_status'], 'unknown')
        self.assertEqual(result['pointwise_robustness'], 'unavailable')

    def test_compute_pointwise_robustness_inf(self):
        vf = LPVerifier()
        result = vf.verify_epsilon_robustness(
            self.network, self.x_0, self.epsilon)
        self.assertEqual(result['problem_status'], 'infeasible')
        self.assertEqual(result['optimal_value'], np.inf)
        self.assertIsNone(result['adversarial_example'])

        self.assertEqual(result['verification_status'], 'unverified')
        self.assertEqual(result['robustness_status'], 'unknown')
        self.assertEqual(result['pointwise_robustness'], 'unavailable')


class TestLPVerifierFeasible(unittest.TestCase):
    W1 = np.array([[1, 0], [0, 1]])
    b1 = np.array([1, 1])
    W2 = np.array([[0, 1], [1, 0]])
    b2 = np.array([2, 2])

    weights = [W1, W2]
    biases = [b1, b2]
    network = Network(weights, biases, activation='ReLU')

    epsilon = 1.5
    x_0 = np.array([1, 2])

    def test_solve_feasible(self):
        vf = LPVerifier()
        vf.network = self.network
        vf.generate_decision_variables()

        obj = cp.Minimize(1)
        result = vf._solve(self.x_0, self.epsilon, obj)
        self.assertEqual(result['problem_status'], 'optimal')
        self.assertEqual(result['optimal_value'], 1.0)
        self.assertAlmostEqual(result['adversarial_example'][0], 1.0134037)
        self.assertAlmostEqual(result['adversarial_example'][1], 0.6628437)

    def test_verify_epsilon_robustness_feasible(self):
        vf = LPVerifier()
        result = vf.verify_epsilon_robustness(
            self.network, self.x_0, self.epsilon)
        self.assertEqual(result['problem_status'], 'optimal')
        self.assertEqual(result['optimal_value'], 1.0)
        self.assertAlmostEqual(result['adversarial_example'][0], 1.0134037)
        self.assertAlmostEqual(result['adversarial_example'][1], 0.6628437)

        self.assertEqual(result['verification_status'], 'verified')
        self.assertEqual(result['robustness_status'], 'not_robust')

    def test_compute_pointwise_robustness_feasible(self):
        vf = LPVerifier()
        result = vf.compute_pointwise_robustness(
            self.network, self.x_0, self.epsilon)
        self.assertEqual(result['problem_status'], 'optimal')
        self.assertAlmostEqual(result['optimal_value'], 0.5)
        self.assertAlmostEqual(result['adversarial_example'][0], 1.5)
        self.assertAlmostEqual(result['adversarial_example'][1], 1.5)

        self.assertAlmostEqual(result['verification_status'], 'verified')
        self.assertAlmostEqual(result['robustness_status'], 'not_robust')
        self.assertAlmostEqual(result['pointwise_robustness'], 0.5)


if __name__ == '__main__':
    unittest.main()
