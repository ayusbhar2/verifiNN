
import cvxpy as cp
import numpy as np

from verifiNN.models.network import Network


class AbstractVerifier:

	def __init__(self, network=None):
		self.network = network
		self.variables = {}
		self.var_name_to_id_map = {}
		self.constraints = []

	def get_var(self, name)-> cp.Variable:
		try:
			var = self.variables[name]
		except KeyError:
			raise('Canot get {}. Variable does not exist!'.format(name))
		
		return var

	def add_var(self, shape: tuple, name: str):
		if len(name) == 0:
			raise ValueError('Cannot create variable with empty name!')

		var = cp.Variable(shape, name=name)
		self.variables[name] = var # check for collision
		self.var_name_to_id_map[name] = var.id

	def extract_solution(self, prob):
		status = prob.solution.status
		opt_val = prob.solution.opt_val
		if not len(prob.solution.primal_vars) == 0:
			z_0_id = self.var_name_to_id_map['z_0'] # hack
			z_0_value = prob.solution.primal_vars[z_0_id]
		else:
			z_0_value = None

		solution = {'problem_status': status,
					'optimal_value': opt_val,
					'adversarial_example': z_0_value}

		return solution	


# TODO: Implement MILP and SDP verifiers
class LPVerifier(AbstractVerifier):

	def generate_decision_variables(self):
		n = len(self.network.weights[0][0]) # num columns in first matrix
		self.add_var(n, 'z_0')

		for i in range(len(self.network.weights)):
			di = len(self.network.weights[i])
			self.add_var(di, f'z_hat_{i+1}')
			self.add_var(di, f'z_{i+1}')
		
	def get_activation_pattern(self, x: np.array) -> list:
		H = self.network.H
		phi = self.network.activation
		weights = self.network.weights
		biases = self.network.biases

		activation_pattern = []
		z = x
		for i in range(H + 1):
			z = list(map(phi, np.dot(weights[i], z) + biases[i]))

			delta = np.array(z)
			delta[delta > 0] = 1
			activation_pattern.append(delta)

		return activation_pattern

	def generate_region_constraints(self, x_0: float, epsilon: float) -> list:
		z_0 = self.get_var('z_0')
		unit = np.ones_like(z_0)
		region_constraints = [z_0 <= x_0 + epsilon * unit,
							  z_0 >= x_0 - epsilon * unit]

		return region_constraints

	def generate_affine_constraints(self) -> list:
		affine_constraints = []
		for i in range(len(self.network.weights)):
			z_hat_i_plus_1 = self.get_var(f'z_hat_{i + 1}')
			z_i = self.get_var(f'z_{i}')
			constr = (z_hat_i_plus_1 == self.network.weights[i] @ z_i + 
										self.network.biases[i])
			affine_constraints.append(constr)

		return affine_constraints

	def generate_ReLU_constraints(self, x_0: float) -> list:
		ReLU_constraints = []
		deltas = self.get_activation_pattern(x_0)

		for i in range(len(deltas)):
			Delta = np.diag(deltas[i])
			n = len(Delta)
			I = np.identity(n)

			z_hat_i_plus_1 = self.get_var(f'z_hat_{i + 1}')
			const1 = ((2 * Delta + I) @ z_hat_i_plus_1 >= 0)
			ReLU_constraints.append(const1)

			z_i_plus_1 = self.get_var(f'z_{i + 1}')
			const2 = (z_i_plus_1 == Delta @ z_hat_i_plus_1)
			ReLU_constraints.append(const2)

		return ReLU_constraints

	def generate_safety_set_constraint(self, l_0: int, l: int) -> bool:
		"""Generate safety set constraint against an adversarial class label."""

		m = len(self.network.weights[-1])
		e_0 = np.eye(1, m, l_0)[0] # unit basis vector with 1 at index l_0
		e_l = np.eye(1, m, l)[0] # unit basis vector with 1 at index l

		z_H_plus_1 = self.get_var(f'z_{self.network.H + 1}')
		ss_cons = ((e_0 - e_l) @ z_H_plus_1 <= 0)

		return ss_cons

	def _solve(self, x_0: np.array, epsilon: float,
		obj: cp.problems.objective.Minimize) -> dict:

		# Constraints
		constraints = []
		region_cons = self.generate_region_constraints(x_0, epsilon) # region const
		aff_cons = self.generate_affine_constraints() # affine const
		ReLU_cons = self.generate_ReLU_constraints(x_0) # ReLU const

		y = self.network.get_output(x_0)
		K = len(y)
		l_0 = np.argsort(y)[-1]

		i = -2
		while i >= -K: # test against each classs
			l = np.argsort(y)[i]
			ss_cons = self.generate_safety_set_constraint(l_0, l) # safety-set const
			constraints = region_cons + aff_cons + ReLU_cons + [ss_cons]

			# Problem
			prob = cp.Problem(obj, constraints)
			prob.solve()

			if prob.status == 'optimal':
				break
			i -= 1

		return self.extract_solution(prob)

	def verify_epsilon_robustness(self, network: Network, x_0: np.array,
		epsilon: float) -> dict:
		"""Verify epsilon-robustness of the network via LP satisfiability."""

		self.network = network
		self.generate_decision_variables()

		obj = cp.Minimize(1)
		result = self._solve(x_0, epsilon, obj)

		if result['problem_status'] == 'optimal':
			result.update({'verification_status': 'verified'})
			result.update({'robustness_status': 'not_robust'})
			result.update({'pointwise_robustness': 'unavailable'})
		else:
			result.update({'verification_status': 'unverified'})
			result.update({'robustness_status': 'unknown'})
			result.update({'pointwise_robustness': 'unavailable'})

		return result

	def compute_pointwise_robustness(self, network: Network, x_0: np.array,
		epsilon: float) -> dict:
		"""Compute the distance to the nearest adversarial example."""

		self.network = network
		self.generate_decision_variables()

		obj = cp.Minimize(cp.norm_inf(x_0 - self.get_var('z_0'))) # hack
		result = self._solve(x_0, epsilon, obj)

		if result['problem_status'] == 'optimal':
			result.update({'verification_status': 'verified'})
			result.update({'robustness_status': 'not_robust'})
			result.update({'pointwise_robustness': result['optimal_value']})
		else:
			result.update({'verification_status': 'unverified'})
			result.update({'robustness_status': 'unknown'})
			result.update({'pointwise_robustness': 'unavailable'})

		return result
