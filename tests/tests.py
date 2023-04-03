import numpy as np
import unittest

from verifiNN.models.network import Network


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
		self.assertTrue((nw.weights[1]== W2).all())
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
		l = network.classify(x)
		self.assertEqual(l, 1)


if __name__ == '__main__':
	unittest.main()

