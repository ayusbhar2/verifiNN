import numpy as np

from utils import calculus, network, numpyutils


np.random.seed(1234)

num_layers = 4  # total layers
dx = 2  # total features
dz = 1  # output vector size
di = 2  # neurons in each layer
H = 1  # number of hidden layers
num_data = 2

arch = network.generate_network_specs(dx, dz, di, H)

# TODO - get input data and z (labels) from the Data class

input_data = numpyutils.get_random_data(1, dx)
z = numpyutils.get_random_data(num_data, dz)


def main():
    # print("starting basic neural network training")
    # print(f"total layers: {H+1}")
    # print(f"input features : {dx}")
    # print(f"neurons in each hidden layer : {di}")
    # print(f"dimensions of output : {dz}")

    w_list = network.initialize_network(arch)
    w_vector = network.unpack_weights(w_list)
    loss = compute_loss(w_vector)
    # print(f"loss after 1 iteration: {loss}")

    gradient_vector = calculus.differentiate(compute_loss, w_vector)
    # print(f"gradient at end of 1 iteration: {gradient_vector}")
    # while(gradient_vector!)


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

    W = np.empty([2, 2])

    size = len(weight_list)
    for i in range(size-1, 0, -1):
        if(i == size-1):
            W = weight_list[i].dot(weight_list[i-1])
        else:
            # print(
            #     f"computing dot of {W} with {weight_list[i-1]}")
            W = W.dot(weight_list[i-1])

    # print(f"final weights: {W}")
    return W


def compute_loss_vector(y, z):
    ''' computes the mean squared distances between 2 vectors (y and z) of equal dimensions (1xn)'''
    x = y-z
    # print(f"y is: {y}")
    # print(f"z is: {z}")
    # print(f"x is: {x}")
    squared_vector = np.square(x)
    # print(f"squared error is: {squared_vector}")
    loss = squared_vector.sum()
    # print(f"loss is {loss}")
    return loss


def compute_loss(weights_vector):
    ''' computes the value of loss function at a given point determined by the weights_vector'''

    W_list = network.pack_weights(w_vector=weights_vector, arch=arch)
    y = compute_output(input_data, W_list)
    return compute_loss_vector(y, z)


main()


# Tests
#y = np.random.randint(10, size=(2, 5))
#z = np.random.randint(10, size=(2, 5))
#compute_loss(y, z)
