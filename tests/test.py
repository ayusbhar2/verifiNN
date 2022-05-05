import numpy as np
import unittest

from utils import calculus


class TestCalculusMethods(unittest.TestCase):

	def test_differentiate_1D(self):
		a = np.array([[2]])
		grad_actual = calculus.differentiate(lambda x: x**2, a)
		grad_expt = ((2.0001)**2 - (1.9999)**2)/0.0002
		self.assertEqual(grad_actual, grad_expt)

	def test_differentiate_2D(self):
		def f(x):
			return x[0]**2 + x[1]**2

		a = np.array([1, 2])
		grad_expt = np.array([2.0, 4.0])
		grad_actual = calculus.differentiate(f, a)
		self.assertAlmostEqual(grad_expt[0],grad_actual[0], delta=0.0001)
		self.assertAlmostEqual(grad_expt[1],grad_actual[1], delta=0.0001)

if __name__ == '__main__':
    unittest.main()