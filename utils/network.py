import numpy as np

# TODO: maybe we can create an `Network` class later.


def generate_network_specs(dx, dz, di, H):
    """Generate a list of tuples contianing matrix dimensions.

    Args:
            dx (Int): length of the input vector
            dz (Int): length of output vector
            di (Int): number of neurons in each hiddeen layer
            H (Int): number of hidden layers (excluding the output layer)
    Returns:
            network_specs (list of tuples): each entry in network_specs is
                    a tuple of the form (num_rows, num_cols). The ith tuple corresponds
                    to the ith weight matrix.
    """
    network_specs = []
    i = 0
    while i < H + 1:
        if i == 0:
            t = (di, dx)
        elif i == H:
            t = (dz, di)
        else:
            t = (di, di)
        network_specs.append(t)
        i += 1
    return network_specs


def initialize_network(network_specs):
    """Generate random weight matrices from network_specs."""
    network = []
    for dims in network_specs:
        W = np.random.rand(dims[0], dims[1]) + 5
        network.append(W)
    return network


def unpack_weights(W_List):
    '''Returns a 1D array by unpacking all weights in the weight list'''
    flat_list = [w.ravel() for w in W_List]
    unpacked_array = np.concatenate(flat_list, axis=0)
    return unpacked_array


def pack_weights(w_vector, arch):
    '''Creates a list of weight matrices form a weights vector accordig to arch.'''

    weight_list = []
    i = 0

    for shape in arch:
        size = shape[0]*shape[1]
        w = w_vector[i:size+i]
        w_reshaped = w.reshape(shape[0], shape[1])
        weight_list.append(w_reshaped)
        i = size

    return weight_list
