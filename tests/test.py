import numpy as np
import unittest

from utils import calculus, network

tol = 0.000001	# tol = O(h^2)

class TestCalculusMethods(unittest.TestCase):


	def test_differentiate_k1_n1(self):
		a = np.array([[2]])
		grad_actual = calculus.differentiate(lambda x: x**2, a)
		grad_expt = ((2.0001)**2 - (1.9999)**2)/0.0002
		self.assertEqual(grad_actual, grad_expt)


	def test_differentiate_k1_n2(self):
		def f(x):
			return x[0]**2 + x[1]**2

		a = np.array([1, 2])
		grad_expt = np.array([2.0, 4.0])
		grad_actual = calculus.differentiate(f, a)
		self.assertAlmostEqual(grad_expt[0],grad_actual[0], delta=tol)
		self.assertAlmostEqual(grad_expt[1],grad_actual[1], delta=tol)


	def test_differentiate_k2_n1(self):
		def f(x):
			return x**3 + 2*x**2 - x + 1

		a = np.array([[-1.0]])
		grad_expt = np.array([-2.0])
		grad_actual = calculus.differentiate(f, a, k=2)
		self.assertAlmostEqual(grad_expt[0], grad_actual[0], delta=tol)

	def test_differentiate_k2_n2(self):
		def f(x):
			return x[0]**3 + 3*x[0]*x[1]**2 - 2*x[1] + 1
		a = np.array([-1.0, 2.0])
		H_expt = np.array([[-6, 12], [12, -6]])
		H_actual = calculus.differentiate(f, a, k=2)
		self.assertAlmostEqual(H_expt[0,0], H_actual[0, 0], delta=tol)
		self.assertAlmostEqual(H_expt[0,1], H_actual[0, 1], delta=tol)
		self.assertAlmostEqual(H_expt[1,0], H_actual[1, 0], delta=tol)
		self.assertAlmostEqual(H_expt[1,1], H_actual[1, 1], delta=tol)

		a = np.array([-0.5, 0.25])
		H_expt = np.array([[-3, 1.5], [1.5, -3]])
		H_actual = calculus.differentiate(f, a, k=2)
		self.assertAlmostEqual(H_expt[0,0], H_actual[0, 0], delta=tol)
		self.assertAlmostEqual(H_expt[0,1], H_actual[0, 1], delta=tol)
		self.assertAlmostEqual(H_expt[1,0], H_actual[1, 0], delta=tol)
		self.assertAlmostEqual(H_expt[1,1], H_actual[1, 1], delta=tol)


class TestNetworksMethods(unittest.TestCase):

	def test_generate_network_specs(self):
		arch = network.generate_network_specs(4, 3, 2, 2)
		self.assertEqual(arch, [(2, 4), (2, 2), (3, 2)])


	def test_initialize_network(self):
		ntwk = network.initialize_network([(2, 4), (2, 2), (3, 2)])
		self.assertEqual(ntwk[0].shape, (2, 4))
		self.assertEqual(ntwk[1].shape, (2, 2))
		self.assertEqual(ntwk[2].shape, (3, 2))



if __name__ == '__main__':
    unittest.main()