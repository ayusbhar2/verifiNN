import numpy as np

num_layers = 4  # total layers
num_features = 2  # total features


def main():
    print("starting basic neural network training")
    print(f"total layers: {num_layers}")
    print(f"input features : {num_features}")

    input_data = get_training_data(100, num_features)
    W = init_weights_network(num_layers, num_features)

    # following needs to be run a loop or matric multiplication - TODO
    output = compute_nn_output_one_iteration(input_data, W)
    print(f"output after 1 iteration: {output}")


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


def compute_nn_output_one_iteration(input_data, weight_list):
    '''computes and returns y for single data point'''
    print(f"starting nn for 1 iteration with input {input_data}")
    print(f"weights:{weight_list}")
    y = compute_output(input_data, weight_list)
    return y


def compute_output(input_data, weight_list):
    '''computes wn.w(n-1)..w1.(input_data)'''

    W = compute_weight_multiplication(weight_list=weight_list)
    output = W.dot(input_data.T)

    return output


def compute_weight_multiplication(weight_list):
    '''computes wn.w(n-1)..w1, where w(i) is the weight matrix of ith layer'''

    multiplied_weights = np.empty([2, 2])

    size = len(weight_list)
    for i in range(size-1, 0, -1):
        if(i == size-1):
            multiplied_weights = weight_list[i].dot(weight_list[i-1])
        else:
            print(
                f"computing dot of {multiplied_weights} with {weight_list[i-1]}")
            multiplied_weights = multiplied_weights.dot(weight_list[i-1])

    print(f"final weights: {multiplied_weights}")
    return multiplied_weights


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
