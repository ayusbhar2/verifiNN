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
