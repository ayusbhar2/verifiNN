from functools import partial

import numpy as np

from verifiNN.utils import calculus


def gradient_descent(model, X, Z, e_t=0.001, alpha=0.05, max_iters=100,
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

    loss = model.compute_loss(X, Z)

    # create a partial object to facilitate differentiation
    compute_loss_partial = partial(model.compute_loss, X, Z)

    gradient_vector = calculus.differentiate(compute_loss_partial, model.params)
    gradient_norm = np.linalg.norm(gradient_vector)

    i = 0
    while(gradient_norm > e_t and i < max_iters):
        # move to new location
        model.params = model.params - alpha*gradient_vector

        # compute gradient at new location
        gradient_vector = calculus.differentiate(compute_loss_partial, model.params)
        gradient_norm = np.linalg.norm(gradient_vector)

        # compute new loss
        loss = model.compute_loss(X=X, Z=Z)

        i = i+1

    return model









