
import numpy as np

# TODO: maybe we can create an `Network` class later.


class Network:

    def __init__(self, dx, dz, di, H):

        self.dx = dx
        self.dz = dz
        self.di = di
        self.H = H
        self.network_specs = []
        self.network = []

    def generate_network_specs(self):
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
        i = 0
        while i < self.H + 1:
            if i == 0:
                t = (self.di, self.dx)
            elif i == self.H:
                t = (self.dz, self.di)
            else:
                t = (self.di, self.di)
            self.network_specs.append(t)
            i += 1
        return self.network_specs

    def initialize_network(self):
        """Generate random weight matrices from network_specs."""
        for dims in self.network_specs:
            W = np.random.rand(dims[0], dims[1])
            self.network.append(W)
        return self.network

    def unpack_weights(self, W_List):
        '''Returns a 1D array by unpacking all weights in the weight list'''
        flat_list = [w.ravel() for w in W_List]
        unpacked_array = np.concatenate(flat_list, axis=0)
        return unpacked_array

    def pack_weights(self, w_vector):
        '''Creates a list of weight matrices form a weights vector accordig to arch.'''

        weight_list = []
        i = 0

        for shape in self.network_specs:
            size = shape[0]*shape[1]
            w = w_vector[i:size+i]
            w_reshaped = w.reshape(shape[0], shape[1])
            weight_list.append(w_reshaped)
            i = size

        return weight_list
