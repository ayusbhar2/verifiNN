
import numpy as np

from verifiNN.utils.common_utils import check_1D_array
from verifiNN.models.model import Model


class LinearRegression(Model):

    def __init__(self):
        self._params = None
        self.trained = False

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, value):
        check_1D_array(value)
        self._params = value

    def initialize(self, n_params=0, offset=0, params=None):
        if params is None:
            self.params = np.random.rand(n_params) + offset
        else:
            self.params = params

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

    def compute_loss(self, X, Z):
        """Compute the loss incurred by the model over a given dataset (X, Z).
        
        X (1D or 2D numpy arrray): m x n matrix with each row an input vector
        Z (1D numpy array): m x 1 matrix with each row a scalar observation
        """
        loss = 0
        for i in range(len(X)):
            y = self.get_output(X[i])
            z = Z[i]
            loss += np.linalg.norm(y - z)**2
        return 0.5*loss


   