import logging
import numpy as np
import unittest

from math import e, log

from verifiNN.algorithms.algorithm import gradient_descent
from verifiNN.models.models import LinearRegression, LogisticRegression
from verifiNN.trainer import TrainingTask
from verifiNN.utils import calculus, common_utils

tol = 0.000001	# tol = O(h^2)


class TestAlgorithm(unittest.TestCase):


	def test_gradient_descent(self):
		X = np.array([[1, 2], [3, 4]])
		Z = np.array([5, 6])
		p = np.array([1, 1, 1])

		lr = LinearRegression()
		lr.initialize(params=p)
		trained_model = gradient_descent(
			lr, X, Z, epsilon=0.001, alpha=0.5, max_iters=0)
		self.assertAlmostEqual(trained_model.params[0], 1)
		self.assertAlmostEqual(trained_model.params[1], 1)
		self.assertAlmostEqual(trained_model.params[2], 1, places=3)

		lr = LinearRegression()
		lr.initialize(params=p)
		trained_model = gradient_descent(
			lr, X, Z, epsilon=0.001, alpha=0.05, max_iters=1)
		self.assertAlmostEqual(trained_model.params[0], 0.95)
		self.assertAlmostEqual(trained_model.params[1], 0.75)
		self.assertAlmostEqual(trained_model.params[2], 0.7, places=3)


class TestLinearRegression(unittest.TestCase):

	def test_params(self):
		lr = LinearRegression()
		self.assertIsNone(lr.params)

		with self.assertRaises(TypeError):
			lr.params = np.array([[1, 2, 3]])

	def test_initialize(self):
		lr = LinearRegression()
		lr.initialize(n_params=2, offset=1)
		self.assertGreater(lr.params[0], 1)
		self.assertLess(lr.params[0], 2)

		self.assertGreater(lr.params[1], 1)
		self.assertLess(lr.params[1], 2)

	def test_get_output(self):
		x = np.array([2])
		p = np.array([1, 2])
		lr = LinearRegression()
		lr.initialize(params=p)

		out = lr.get_output(x)
		self.assertEqual(out, 5)

		with self.assertRaises(TypeError):
			x = [2] # x should be a np 1D arrray
			out = lr.get_output(x)

	def test_compute_loss(self):
		X = np.array([[1, 2], [3, 4]])
		Z = np.array([5, 6])
		p = np.array([1, 1, 1])

		lr = LinearRegression()
		lr.initialize(params=p)
		loss = lr.compute_loss(X, Z)
		self.assertEqual(loss, 2.5)


class TestLogisticRegression(unittest.TestCase):

	def test_get_output(self):
		x = np.array([1, 2])
		p = np.array([1, 1, 1])

		lor = LogisticRegression()
		lor.initialize(params=p)
		out = lor.get_output(x)

		self.assertEqual(out, 0.9820137900379085)

	def test_compute_loss(self):
		X = np.array([[1, 2], [3, 4]])
		Z = np.array([0, 1])
		p = np.array([1, 1, 1])

		lor = LogisticRegression()
		lor.initialize(params=p)
		loss = lor.compute_loss(X, Z)
		exp_loss = -log(1 - e**4 / (1 + e**4)) - log(e**8 / (1 + e**8))

		self.assertEqual(loss, 4.018485334290706)




# class TestNetwork(unittest.TestCase):


# 	def test_generate_network_specs(self):
# 		network = Network(4, 3, 2, 2)
# 		arch = network.generate_network_specs()

# 		self.assertEqual(arch, [(2, 4), (2, 2), (3, 2)])


# 	def test_initialize_network(self):
# 		network = Network(4, 3, 2, 2)
# 		network.generate_network_specs()
# 		ntwk = network.initialize_network()

# 		self.assertEqual(ntwk[0].shape, (2, 4))
# 		self.assertEqual(ntwk[1].shape, (2, 2))
# 		self.assertEqual(ntwk[2].shape, (3, 2))



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


class TestCommonUtils(unittest.TestCase):
	def test_unpack_weights(self):
		W1 = np.array([[1, 2], [3, 4]])
		W2 = np.array([[5, 6], [7, 8]])
		W_list = [W1, W2]
		w = common_utils.unpack_weights(W_list)
		self.assertTrue((w==[1, 2, 3, 4, 5, 6, 7, 8]).all())

	def test_pack_weights(self):
		w = np.array([1, 2, 3, 4, 5, 6, 7, 8])
		W_list = common_utils.pack_weights(w, [(2, 2), (2, 2)])
		self.assertTrue(
			(W_list[0]==np.array([[1, 2], [3, 4]])).all()
		)
		self.assertTrue(
			(W_list[1]==np.array([[5, 6], [7, 8]])).all()
		)

	def test_logistic(self):
		self.assertAlmostEqual(
			common_utils.logistic(1), 0.7310585786300049)


# class TestController(unittest.TestCase):
# 	def test_single_data_origin(self):
# 		# print("starting test --- test_single_data_origin")
# 		dx = 2  # total features
# 		dz = 1  # output vector size
# 		di = 2  # neurons in each layer
# 		H = 1  # number of hidden layers
		
# 		input_data = np.array([0.53436063, 0.31448347])
		
# 		#z = np.array([[0.18450251],[0.28731715]])
# 		z = np.array([0.18450251])
				  
# 		network = Network(dx, dz, di, H)
		
# 		task = TrainingTask(model=network, input_data=input_data, z=z)
# 		task.initialize()
# 		output = task.start_training(
# 			algorithm_function=gradient_descent,
# 			max_iterations=200, verbose=False)
# 		# Validate norm less that eta
# 		self.assertLess(output[1],0.005)

# 	def test_single_data_5(self):
# 		# print("starting test --- test_single_data_5")
# 		dx = 2  # total features
# 		dz = 1  # output vector size
# 		di = 2  # neurons in each layer
# 		H = 1  # number of hidden layers
		
# 		input_data = np.array([0.53436063, 0.31448347])
		
# 		z = np.array([0.18450251])
				  
# 		network = Network(dx, dz, di, H)
		
# 		task = TrainingTask(model=network, input_data=input_data, z=z)
# 		task.initialize(5)
# 		output = task.start_training(
# 			algorithm_function=gradient_descent,
# 			verbose=False)
# 		# Validate norm less that eta
# 		self.assertLess(output[1],0.005)



if __name__ == '__main__':
    unittest.main()
