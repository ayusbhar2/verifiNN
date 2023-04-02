import numpy as np
from math import e

def unpack_weights(W_List):
    '''Returns a 1D array by unpacking all weights in the weight list'''
    flat_list = [w.ravel() for w in W_List]
    unpacked_array = np.concatenate(flat_list, axis=0)
    return unpacked_array


def pack_weights(w_vector, architecture):
    '''Creates a list of weight matrices form a weights vector accordig to arch.'''

    weight_list = []
    i = 0

    for shape in architecture:
        size = shape[0]*shape[1]
        w = w_vector[i:size+i]
        w_reshaped = w.reshape(shape[0], shape[1])
        weight_list.append(w_reshaped)
        i = size

    return weight_list

def check_1D_array(value):
    if (not type(value) == np.ndarray) or (len(value.shape) != 1):
        raise TypeError(
            "expecting a 1D numpy array, got {}".format(type(value))
        )

def logistic(x):
    return e**(x)/(1 + e**x)