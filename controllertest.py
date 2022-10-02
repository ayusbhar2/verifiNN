import unittest

import numpy as np

from TrainingTask import TrainingTask
from utils import algorithm
from utils.network import Network


class TestControllerNN(unittest.TestCase):

	verbose=False

	def test_single_data_origin(self):
		print("starting test --- test_single_data_origin")
		dx = 2  # total features
		dz = 1  # output vector size
		di = 2  # neurons in each layer
		H = 1  # number of hidden layers
		
		input_data = np.array([0.53436063, 0.31448347])
		
		#z = np.array([[0.18450251],[0.28731715]])
		z = np.array([0.18450251])
				  
		network = Network(dx, dz, di, H)
		
		task = TrainingTask(network=network, input_data=input_data, z=z)
		task.initialize()
		output = task.start_training(
			algorithm_function=algorithm.gradient_descent,
			verbose=TestControllerNN.verbose)
		# Validate norm less that eta
		self.assertLess(output[1],0.005)

	def test_single_data_5(self):
		print("starting test --- test_single_data_5")
		dx = 2  # total features
		dz = 1  # output vector size
		di = 2  # neurons in each layer
		H = 1  # number of hidden layers
		
		input_data = np.array([0.53436063, 0.31448347])
		
		z = np.array([0.18450251])
				  
		network = Network(dx, dz, di, H)
		
		task = TrainingTask(network=network, input_data=input_data, z=z)
		task.initialize(5)
		output = task.start_training(
			algorithm_function=algorithm.gradient_descent,
			verbose=TestControllerNN.verbose)
		# Validate norm less that eta
		self.assertLess(output[1],0.005)

	def test_double_data_origin(self):
		print("starting test --- test_double_data_origin")
		dx = 2  # total features
		dz = 1  # output vector size
		di = 2  # neurons in each layer
		H = 1  # number of hidden layers
		
		input_data = np.array([[0.53436063, 0.31448347],
							   [0.73436063, 0.81448347]])
		
		z = np.array([[0.18450251],
                      [0.28731715]])
				  
		network = Network(dx, dz, di, H)
		
		task = TrainingTask(network=network, input_data=input_data, z=z)
		task.initialize()
		output = task.start_training(
			algorithm_function=algorithm.gradient_descent,
			alpha=0.5, max_iterations=300,verbose=TestControllerNN.verbose)
		# Validate norm less that eta
		self.assertLess(output[1],0.005)





if __name__ == '__main__':
    unittest.main()
