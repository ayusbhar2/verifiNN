import numpy as np

def ReLU(x):
	if x > 0:
		return x
	else:
		return 0

def Id(x):
	return x

ACTIVATION_FUNCTIONS = {'ReLU': ReLU, 'Id': Id}
LABELING_FUNCTIONS = {'argmax': np.argmax}


class Network:

	def __init__(self, weights, biases, activation='Id', labeler='argmax'):
		self.weights = weights
		self.biases = biases
		self.activation = ACTIVATION_FUNCTIONS[activation]
		self.labeler = LABELING_FUNCTIONS[labeler]
		self.H = len(weights) - 1 # num of hidden layers

	def get_output(self, x):
		res = x
		for i in range(self.H + 1):
			res = list(map(
				self.activation, np.dot(self.weights[i], res) + self.biases[i]))

		return res

	def classify(self, x):
		y = self.get_output(x)
		return self.labeler(y)
