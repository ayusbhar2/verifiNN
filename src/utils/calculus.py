import numpy as np

def differentiate(f, a, k=1, h=.0001):
	"""Numerically find the value of the kth derivative of f at a.

	Args:
	    f: function whose derivative is to be computed
	    a: the point at which the derivative is to be computed
	    k: order of the derivative to be computed (upto 2)
	    h: the inifinitesimal to be used in computing the difference quotient
	Returns:
	    f^(k)(a): vector if k=1, matrix if k=2.

	NOTE: Assumes that f is twice differentiable and therefore the Hessian is
		  symmetric.
	"""
	n = len(a)
	grad = np.zeros(n)
	H = np.zeros([n, n])
	I = np.identity(n, dtype=float)

	if k == 1:	# re
		for i in range(n):
			v = np.zeros(n)
			v[i] = h
			a_r = a + v
			a_l = a - v
			q = (f(a_r) - f(a_l))/(2*h)
			grad[i] = q
		return grad

	elif k == 2:
		for i in range(n):
			for j in range(n):
				sum = 0.0
				for s in range(2):
					for t in range(2):
						sum += (-1)**(s+t) * f(a + h * ((-1)**s * I[i, :] + (-1)**t * I[j, :]))
				H[i, j] = 1/(2*h)**2 * sum	# TODO: reduce ops, Hessian is symmetric.
		return H
	else:
		raise(ValueError("Order k > 2 not supported."))



