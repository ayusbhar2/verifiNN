

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
	# assumes func has a derivative or the order
	q = (f(a+h) - f(a-h))/(2*h)
	return q



