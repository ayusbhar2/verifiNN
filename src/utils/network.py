
import numpy as np

from src.utils import commonutils
from src.utils.model import Model


class Network(Model):

    def __init__(self, dx, dz, di, H):

        self.dx = dx
        self.dz = dz
        self.di = di
        self.H = H
        self.network_specs = []
        self.network = []
        self.trained = False;

    def initialize(self, start=0):
        self.generate_network_specs()
        w_list = self.initialize_network(start)
        self.initial_weights = self.unpack_weights(w_list)
        return self.initial_weights


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

    def initialize_network(self, start=0):
        """Generate random weight matrices from network_specs."""
        for dims in self.network_specs:
            W = np.random.rand(dims[0], dims[1]) + start
            self.network.append(W)
        return self.network

    def compute_loss(self,weights_vector, input_data, z):
        ''' computes the value of loss function at a given point determined by the weights_vector'''
        W_list = commonutils.pack_weights(
            w_vector=weights_vector, list_shape_tuple=self.network_specs)
        y = self.get_output(input_data, W_list)
        #print(f"output: {y} ")
        return commonutils.mean_square_distance(y, z)

    def unpack_weights(self, W_List):
        '''Returns a 1D array by unpacking all weights in the weight list'''
        return commonutils.unpack_weights(W_List=W_List)

    def pack_weights(self, w_vector):
        '''Creates a list of weight matrices form a weights vector accordig to arch.'''
        return commonutils.pack_weights(w_vector=w_vector, list_shape_tuple=self.network_specs)

    def get_output(self,input_data, weight_list):
        '''computes wn.w(n-1)..w1.(input_data)'''
        '''computes and returns y for single data point'''

        # print(f"starting nn for 1 iteration with input {input_data}")
        # print(f"weights:{weight_list}")

        W = self.compute_weight_multiplication(weight_list=weight_list)
        output = W.dot(input_data.T)

        return output

    def get_trained_output(self, input_data):
        # TODO - to be implemented
        return super().get_trained_output(input_data)

    def is_trained(self):
        return self.trained;


    def compute_weight_multiplication(self,weight_list):
        '''computes wn.w(n-1)..w1, where w(i) is the weight matrix of ith layer'''

        W = np.empty([2, 2])  # TODO - any other way to initialize?

        size = len(weight_list)
        for i in range(size-1, 0, -1):
            if(i == size-1):
                W = weight_list[i].dot(weight_list[i-1])
            else:
                W = W.dot(weight_list[i-1])

        # print(f"final weights: {W}")
        return W


