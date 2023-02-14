import numpy as np


def unpack_weights(W_List):
    '''Returns a 1D array by unpacking all weights in the weight list'''
    flat_list = [w.ravel() for w in W_List]
    unpacked_array = np.concatenate(flat_list, axis=0)
    return unpacked_array


def pack_weights(w_vector, list_shape_tuple):
    '''Creates a list of weight matrices form a weights vector accordig to arch.'''

    weight_list = []
    i = 0

    for shape in list_shape_tuple:
        size = shape[0]*shape[1]
        w = w_vector[i:size+i]
        w_reshaped = w.reshape(shape[0], shape[1])
        weight_list.append(w_reshaped)
        i = size

    return weight_list

def mean_square_distance(y, z):
    ''' computes the mean squared distances between 2 vectors (y and z) of equal dimensions (1xn)'''
    x = y-z
    squared_vector = np.square(x)
    loss = squared_vector.sum()
    return loss
