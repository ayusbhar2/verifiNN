import numpy as np

def differentiate(f, a, k=1, h=.0001):
	"""Numerically find the value of the kth derivative of f at a.

	Args:
	    f: function whose derivative is to be computed
	    a: the point at which the derivative is to be computed
	    k: order of the derivative to be computed
	    h: the inifinitesimal to be used in computing the difference quotient
	Returns:
	    f^(k)(a)
	"""
	# assumes derivative exists
	n = len(a)
	grad = np.zeros(n)
	for i in range(n):
		v = np.zeros(n)
		v[i] = h
		a_r = a + v
		a_l = a - v
		q = (f(a_r) - f(a_l))/(2*h)
		grad[i] = q
	return grad



