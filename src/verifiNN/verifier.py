
import cvxpy as cp
import numpy as np

from verifiNN.models.network import Network


class Verifier:

	def __init__(self, network: Network, epsilon: float, x0: np.array):
		self.network = network
		self.epsilon = epsilon
		self.x0 = x0


	def get_activation_patterns(self, x: np.array):
		H = self.network.H
		phi = self.network.activation
		weights = self.network.weights
		biases = self.network.biases

		activation_patterns = []
		z = x
		for i in range(H + 1):
			z = list(map(phi, np.dot(weights[i], z) + biases[i]))

			delta = np.array(z)
			delta[delta > 0] = 1 # only works for networks with ReLU
			activation_patterns.append(delta)

		return activation_patterns


	def verify(self):
		classes = range(len(self.network.weights[-1]))
		y = self.network.get_output(self.x0)

		idx_y = np.argsort(y) # TODO: use the network's labeler function
		l0 = idx_y[-1]
		l1 = idx_y[-2] # TODO: implement for all K - 1 class labels










		