import logging
import numpy as np

from functools import partial
from verifiNN.utils import calculus

#TODO(ayush): Make class so that all algorithms have a consistent interface

def gradient_descent(model, X, Z, epsilon=0.001, alpha=0.05, max_iters=100,
    verbose=False):
    """Train a model with gradient descent.

    Inputs:
        model(object): Object of a subclass of the Model class
        X (1D or 2D np array): m x n matrix with each row an example
        Z (1D or 2D np array): m x p matrix with each row an observation
        e_t(float): Epsilon threshold. We want the magnitude of the gradient
                    vector to be within this threshold.
        alpha(float): learning rate
        max_iters(int): maximum number of iterations to run.
    """
    i = 0
    logging.info("starting gradient descent...")

    # create a partial object to facilitate differentiation
    compute_loss_partial = partial(model.compute_loss, X, Z)

    gradient_vector = calculus.differentiate(compute_loss_partial, model.params)
    gradient_norm = np.linalg.norm(gradient_vector)
    loss = model.compute_loss(X, Z)

    logging.info("<iteration>, <loss>, <gradient_norm>")
    logging.info("{}, {}, {}".format(i, loss, gradient_norm))

    while(gradient_norm > epsilon and i < max_iters):
        # move to new location
        model.params = model.params - alpha*gradient_vector

        # compute gradient at new location
        gradient_vector = calculus.differentiate(compute_loss_partial, model.params)
        gradient_norm = np.linalg.norm(gradient_vector)
        loss = model.compute_loss(X=X, Z=Z)
        i += 1

        logging.info("{}, {}, {}".format(i, loss, gradient_norm))

    return model









