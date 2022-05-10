import numpy as np

<<<<<<< HEAD
num_layers = 2  # total layers
=======
np.random.seed(1234)

num_layers = 4  # total layers
>>>>>>> 714f74afed91f627e993a425e5640dbd64a48746
num_features = 2  # total features


def main():
    print("starting basic neural network training")
    print(f"total layers: {num_layers}")
    print(f"input features : {num_features}")

    input_data = get_training_data(1, num_features)
    W_list = init_weights_network(num_layers, num_features)

    output = compute_output(input_data, W_list)
    print(f"output after 1 epic: {output}")


def get_training_data(rows, columns):
    return np.random.randint(5, size=(rows, columns))


def init_weights_network(num_layers, num_features):
    '''initializes weights for the whole network and returns a list of weight matrix'''
    W = []
    for i in range(num_layers-1):
        # fully connected
        W.append(initialize_weights(num_features, num_features))

    # last layer - assuming scalar output per iteration
    W.append(initialize_weights(1, num_features))
    return W


def initialize_weights(num_neuron, num_features):
    return np.random.uniform(0.0, 1.0, size=(num_neuron, num_features))


def compute_output(input_data, weight_list):
    '''computes wn.w(n-1)..w1.(input_data)'''
    '''computes and returns y for single data point'''

    print(f"starting nn for 1 iteration with input {input_data}")
    print(f"weights:{weight_list}")

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
            print(
                f"computing dot of {W} with {weight_list[i-1]}")
            W = W.dot(weight_list[i-1])

    print(f"final weights: {W}")
    return W


def compute_loss(y, z):
    ''' computes the mean squared distances between 2 vectors (y and z) of equal dimensions (1xn)'''
    x = y-z
    print(f"y is: {y}")
    print(f"z is: {z}")
    print(f"x is: {x}")
    squared_vector = np.square(x)
    print(f"squared error is: {squared_vector}")
    loss = squared_vector.sum(axis=1)
    print(f"loss is {loss}")
    return loss


main()


# Tests
y = np.random.randint(10, size=(2, 5))
z = np.random.randint(10, size=(2, 5))
compute_loss(y, z)
