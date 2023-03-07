import logging

from verifiNN.models.models import Model


class TrainingTask:

    def __repr__(self):
        return ""

    def __init__(self, model: Model, X, Z, algorithm, epsilon=0.001, alpha=0.05,
        max_iters=100, verbose=False):
        """
        Inputs:
            model(Model): Object of base class Model

            X(numpy ndarray): m x n matrix with each row a training example
                and each column a feature.

            Z(numpy ndarray): m x p matrix with each row an observation. For
                scalar observations, p = 1.
            algorithm(function): Algorithm to use for loss minimization

            epsilon(float): Max acceptable norm of the gradient vector. epsilon
                represents the acceptable distance from a critical point where
                the norrm of the gradient will be exacetly 0.

            aplha(float): Learning rate

            max_iters(int): Maximum number of iterations to run

        Output: None

        """
        self.model = model
        self.X = X
        self.Z = Z
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.alpha = alpha
        self.max_iters = max_iters
        self.verbose = verbose

    def start(self):
        logging.info("-------------start training-------------")
        logging.info("model: {}".format(self.model.__class__))
        logging.info("algorithm: {}".format(self.algorithm))
        logging.info("epsilon: {}".format(self.epsilon))
        logging.info("alpha: {}".format(self.alpha))
        logging.info("max_iters: {}".format(self.max_iters))

        self.algorithm(self.model, self.X, self.Z, epsilon=self.epsilon,
            alpha=self.alpha, max_iters=self.max_iters, verbose=self.verbose)

        
