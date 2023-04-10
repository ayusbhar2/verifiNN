import numpy as np

from verifiNN.utils.utils import ACTIVATION_FUNCTIONS, LABELING_FUNCTIONS

class Network:

	def __init__(self, weights, biases, activation='Id', labeler='argmax'):
		self.weights = weights
		self.biases = biases
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.labeler = LABELING_FUNCTIONS[labeler]
		self.H = len(weights) - 1 # num of hidden layers

	def get_output(self, x):
		result = x
		for i in range(self.H + 1):
			result = list(map(
				self.activation, np.dot(self.weights[i], result) + self.biases[i]))

		return result

	def classify(self, x):
		y = self.get_output(x)
		return self.labeler(y)
