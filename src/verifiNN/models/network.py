
import numpy as np

from verifiNN.utils import common_utils
from verifiNN.models.models import Model


def ReLU(x):
    if x > 0:
        return x
    else:
        return 0

def Id(x):
    return x


ACTIVATION_FUNCTIONS = {'ReLU': ReLU, 'Id': Id}


class Network(Model):
    """Class representing a Fully-Connected-Feed-Forward Neural Network."""

    def __init__(self, dx=None, dy=None, num_hidden_neurons=None, activation='Id'):
        """Specify the architechture and activation of the network.

        Args:
            dx (int): length of the input vector
            dy (int): length of output vector
            num_hidden_neurons (list of ints): number of neurons in each hidden
                layer (excluding the output layer). `num_hidden_neurons` is a list
                of integers with the value at the i-th position representing
                the number of neurons in the i-th hidden layer.
        """
        self.dx = dx
        self.dy = dy
        if num_hidden_neurons is not None:
            if not isinstance(num_hidden_neurons, list):
                raise("Expecting num_hidden_neurons to be a list. Found {}".format(
                    type(num_hidden_neurons)))
        self.num_hidden_neurons = num_hidden_neurons
        self.activation = ACTIVATION_FUNCTIONS[activation]
        self.weights = []
        self.biases = []

    
    def initialize(self, start_offset=0, weights=None, biases=None):
        """Initialize network using random or user-supplied weights."""

        if weights is not None and biases is not None:
            # Weights were explicitly provided.
            self.weights = weights
            self.biases = biases
            self.dx = weights[0].shape[1]
            self.dy = weights[-1].shape[0]
            self.num_hidden_neurons = [w.shape[0] for w in weights[:-1]]
            self.H = len(self.num_hidden_neurons)

        else:
            # Weights were not provided. Random initialize.
            if (self.dx is None or
                self.dy is None or
                self.num_hidden_neurons is None):
                raise("dx, dy and num_hidden_neurons must be provided!")

            H = len(self.num_hidden_neurons) # number of hidden layers
            for i in range(H + 1):
                if i == 0:
                    m = self.num_hidden_neurons[i]
                    n = self.dx
                elif i == H:
                    m = self.dy
                    n = self.num_hidden_neurons[i - 1]
                else:
                    m = self.num_hidden_neurons[i]
                    n = self.num_hidden_neurons[i - 1]
                W = np.random.rand(m, n) + start_offset
                b = np.random.rand(m) + start_offset

                self.weights.append(W)
                self.biases.append(b)
                self.H = H


    def get_output(self, x):
        res = x
        for i in range(self.H + 1):
            res = list(map(
                self.activation, np.dot(self.weights[i], res) + self.biases[i]))

        return res


    def compute_loss(self,weights_vector, input_data, z):
        '''Computes the value of loss function at a given point determined by the weights_vector'''
        W_list = common_utils.pack_weights(
            w_vector=weights_vector, list_shape_tuple=self.network_specs)
        y = self.get_output(input_data, W_list)
        #print(f"output: {y} ")
        return common_utils.mean_square_distance(y, z)


