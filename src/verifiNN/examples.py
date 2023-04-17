import numpy as np

from verifiNN.models.network import Network
from verifiNN.verifier import LPVerifier


# Defining a network
W1 = np.array([[1, 0],
			  [0, 1]])
b1 = np.array([1, 1])
W2 = np.array([[0, 1],
			  [1, 0]])
b2 = np.array([2, 2])

weights = [W1, W2]
biases = [b1, b2]
network = Network(weights, biases, activation='ReLU', labeler='argmax')

# Classifying an input with a network
x_0 = np.array([1, 2])
l_0 = network.classify(x_0) # class 0
assert l_0 == 0

x_1 = np.array([2, 1])
l_1 = network.classify(x_1) # class 1
assert l_1 == 1

# Compute the pointwise robustness of a network at a reference point
x_0 = np.array([1, 2]); epsilon = 1.5

vf = LPVerifier()
result = vf.compute_pointwise_robustness(network, x_0, epsilon)
assert result['verification_status'] == 'verified'
assert result['robustness_status'] == 'not_robust'

rho = np.round(result['pointwise_robustness'], decimals=5)
assert rho == 0.5

x_hat = result['adversarial_example']
assert np.round(x_hat[0], decimals=5) == 1.5
assert np.round(x_hat[1], decimals=5) == 1.5

