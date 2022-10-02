from functools import partial

import numpy as np

from utils import calculus, commonutils, network


def gradient_descent(weights_vector, m_input_data, m_z, network, e_t=0.001, alpha=0.05, max_iterations=100, verbose=False):
    print(f"initial weights are: {weights_vector}")

    network_specs = network.network_specs
    # partial
    compute_loss_partial = partial(
        compute_loss, input_data=m_input_data, z=m_z, network_specs=network_specs)

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


def compute_loss(weights_vector, input_data, z, network_specs):
    ''' computes the value of loss function at a given point determined by the weights_vector'''
    W_list = commonutils.pack_weights(
        w_vector=weights_vector, list_shape_tuple=network_specs)
    y = compute_output(input_data, W_list)
    #print(f"output: {y} ")
    return mean_square_distance(y, z)


def compute_output(input_data, weight_list):
    '''computes wn.w(n-1)..w1.(input_data)'''
    '''computes and returns y for single data point'''

    # print(f"starting nn for 1 iteration with input {input_data}")
    # print(f"weights:{weight_list}")

    W = compute_weight_multiplication(weight_list=weight_list)
    output = W.dot(input_data.T)

    return output


def compute_weight_multiplication(weight_list):
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


def mean_square_distance(y, z):
    ''' computes the mean squared distances between 2 vectors (y and z) of equal dimensions (1xn)'''
    x = y-z
    squared_vector = np.square(x)
    loss = squared_vector.sum()
    return loss


