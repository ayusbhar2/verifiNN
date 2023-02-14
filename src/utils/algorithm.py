from functools import partial

import numpy as np

from src.utils import calculus


def gradient_descent(weights_vector, m_input_data, m_z, model, e_t=0.001, alpha=0.05, max_iterations=100, verbose=False):
    print(f"initial weights are: {weights_vector}")

    # partial
    compute_loss_partial = partial(
        model.compute_loss, input_data=m_input_data, z=m_z)

    gradient_vector = calculus.differentiate(
        compute_loss_partial, weights_vector)
    norm = np.linalg.norm(gradient_vector)
    loss = compute_loss_partial(weights_vector=weights_vector)
    print(f"initial norm is: {norm}")
    print(f"initial loss is: {loss}")

    i = 0
    while(norm > e_t and i < max_iterations):
        weights_vector = weights_vector - alpha*gradient_vector
        gradient_vector = calculus.differentiate(
            compute_loss_partial, weights_vector)
        # compute magnitude of gradient vector
        norm = np.linalg.norm(gradient_vector)
        # compare this magnitude with e_t
        loss = compute_loss_partial(weights_vector=weights_vector)
        i = i+1

        if(verbose):
            print(f"loss is: {loss}")
            print(f"norm is: {norm}")

    print(f"final weights are: {weights_vector}")
    print(f"total iterations: {i}")
    print(f"final loss is: {loss}")
    print(f"final norm is: {norm}")
    return (weights_vector,norm)









