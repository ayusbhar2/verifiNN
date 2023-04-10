import numpy as np

from verifiNN.models.network import Network


W1 = np.array([[1, 0],
			  [0, 1]])
b1 = np.array([1, 1])
W2 = np.array([[0, 1],
			  [1, 0]])
b2 = np.array([2, 2])

weights = [W1, W2]
biases = [b1, b2]
network = Network(weights, biases, activation='ReLU', labeler='argmax')

x_0 = np.array([1, 2])
l_0 = network.classify(x_0) # returns 0

x_1 = np.array([2, 1])
l_1 = network.classify(x_1) # returns 1

