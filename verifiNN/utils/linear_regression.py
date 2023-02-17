
import numpy as np

from src.utils import commonutils
from src.utils.model import Model


class LinearRegression(Model):

    def __init__(self, dx):

        self.dx = dx
        self.theta = []
        self.trained = False

    def initialize(self, start = 0):
        return self.initialize_weights(start)

    def is_trained(self):
        return self.trained

    def get_output(self, input_data, weight_list):
        return (weight_list.T).dot(input_data)

    def get_trained_output(self, input_data):
        return (self.theta.T).dot(input_data)
        
         
    def initialize_weights(self, start=0):
        """Generate random weight matrices."""

        self.theta = np.random.rand(1, self.dx) + start
        return self.theta

    def compute_loss(self, weights_vector, input_data, z):
        y = self.get_output(input_data=input_data, weight_list=weights_vector)
        return commonutils.mean_square_distance(y, z)


   