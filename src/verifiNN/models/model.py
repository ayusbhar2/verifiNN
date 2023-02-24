from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract class to represent a model.

    A model is a function parametrized by a set of real/complex numbers called
    parameters. Training a model is the process of finding the optimal values
    of these parameters that minimize some loss function over a given training
    set. A trained model maps points in the domain of the function to points
    in the co-domain of the function.
    """

    # ~ Properties ~ #

    @property
    @abstractmethod
    def params(self):
        """Abstract method invoked when the `params` property of a model is
        accessed.

        Each model will have a `params` property containing the parameters of
        the model. e.g. `params` would contain the weights in the case of a
        neural network and the coefficients in the case of linear regression.

        NOTE: params must be a 1D numpy array.
        """

        pass

    @params.setter
    @abstractmethod
    def params(self, value):
        """Abstract method invoked when the `params` property of a model is
        set.

        NOTE: value must be a 1D numpy array.
        """
        pass

    # @property
    # @abstractmethod
    # def is_trained(self):
    #     """."""
    #     pass

    # @is_trained.setter
    # @abstractmethod
    # def is_trained(self, value):
    #     pass


    # ~ Methods ~ #

    @abstractmethod
    def initialize(self, kwargs={}):
        """Set the starting point of descent in the weight space."""
        pass


    @abstractmethod
    def get_output(self, x):
        """Compute f(x), i.e. the output of the model."""
        pass

    # @abstractmethod
    # def get_trained_output(self,input_data):
    #     pass

    @abstractmethod
    def compute_loss(self, X, Z):
        """Compute the loss incurred by the model over a given dataset (X, Z).
        
        X (1D or 2D numpy arrray): m x n matrix with each row an input vector
        Z (1D or 2D numpy array): m x p matrix with each row an observation
            (scalar or vector)
        """
        pass
      
