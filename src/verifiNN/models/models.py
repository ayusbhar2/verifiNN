import numpy as np

from abc import ABC, abstractmethod
from math import log

from verifiNN.utils.common_utils import check_1D_array, logistic


class Model(ABC):
    """Base class to represent a model.

    A model is a function parametrized by a set of real/complex numbers called
    parameters. Training a model is the process of finding the optimal values
    of these parameters that minimize some loss function over a given training
    set. A trained model maps points in the domain of the function to points
    in the co-domain of the function.
    """

    def __init__(self):
        self._params = None


    def initialize(self, n_params=0, offset=0, params=None):
        if params is None:
            self.params = np.random.rand(n_params) + offset
        else:
            # params were explicitly provided
            self.params = params


    @abstractmethod
    def get_output(self, x):
        """Compute f(x), i.e. the output of the model."""
        pass


    @abstractmethod
    def compute_loss(self, X, Z):
        """Compute the loss incurred by the model over a given dataset (X, Z).
        
        X (1D or 2D numpy arrray): m x n matrix with each row an input vector
        Z (1D or 2D numpy array): m x p matrix with each row an observation
            (scalar or vector)
        """
        pass

    # ~ Properties ~ #

    @property
    def params(self):
        return self._params


    @params.setter
    def params(self, value):
        check_1D_array(value)
        self._params = value

      

class LinearRegression(Model):

    def get_output(self, x):
        """Compute f(x).

        Input:
            x(1D numpy array): Single input vector. It is assumed that
                               len(x) = len(params) - 1
        Output:
            f(x) (float): theta_0 + theta_1 x_1 + ... + theta_n x_n
        """
        check_1D_array(x)
        x_1 = np.append([1], x) # add the affine term
        return np.dot(self.params, x_1)

    def compute_loss(self, X, Z, params=None):
        """Compute the loss incurred by the model over a given dataset (X, Z).
        
        X (2D numpy arrray): m x n matrix with each row an input vector
        Z (1D numpy array): m x 1 matrix with each row a scalar observation
        """

        # TODO(ayush): This is a hack to make `partial` work in algorithm.py
        # Find a cleaner way to do this.
        if params is not None:
            # params were explicitly provided
            self.params = params

        # TODO(ayush): Vectorize the below for loop.
        loss = 0
        for i in range(len(X)):
            y = self.get_output(X[i])
            z = Z[i]
            loss += np.linalg.norm(y - z)**2
        return 0.5*loss


class LogisticRegression(Model):

    def get_output(self, x):
        check_1D_array(x) # TODO(ayush): Don't do this every time
        x_1 = np.append([1], x) # add the affine term
        y = np.dot(self.params, x_1)

        return logistic(y)

    def compute_loss(self, X, Z, params=None):
        """Compute the loss incurred by the model over a given dataset (X, Z).
        
        X (2D numpy arrray): m x n matrix with each row an input vector
        Z (1D numpy array): m x 1 matrix with each row a scalar observation
        """
        if params is not None:
            # params were explicitly provided
            self.params = params
        loss = 0
        for i in range(len(X)):
            y = self.get_output(X[i])
            z = Z[i]
            loss += - z*log(y) - (1 - z)*log(1 - y)

        return loss    

    def classify(self):
        pass

