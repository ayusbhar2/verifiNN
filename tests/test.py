import numpy as np
import unittest

from utils import calculus



class TestCalculusMethods(unittest.TestCase):

	def test_differentiate(self):
		grad_actul = calculus.differentiate(lambda x: x**2, 2)
		grad_expt = ((2.0001)**2 - (1.9999)**2)/0.0002
		self.assertEqual(grad_actul, grad_expt)




if __name__ == '__main__':
    unittest.main()