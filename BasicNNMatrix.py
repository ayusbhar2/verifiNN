import numpy as np

num_layers = 4  # total layers
num_features = 2  # total features


def main():
    print("starting basic neural network training")
    print(f"total layers: {num_layers}")
    print(f"input features : {num_features}")

    input_data = get_training_data()
    W = init_weights_network(num_layers, num_features)

    # following needs to be run a loop or matric multiplication - TODO
    output = compute_nn_output_one_iteration(input_data, W)
    print(f"output after 1 iteration: {output}")


def get_training_data():
    return np.random.randint(5, size=(1, num_features))


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
    return np.random.randint(5, size=(num_neuron, num_features))


def compute_nn_output_one_iteration(input_data, weight_list):
    '''computes and returns y for single data point'''
    print(f"starting nn for 1 iteration with input {input_data}")
    print(f"weights:{weight_list}")
    y = compute_output(input_data, weight_list)
    return y


def compute_output(input_data, weight_list):
    '''computes w1.w2..wn.(input_data)'''
    output = input_data.transpose()
    print(f"output: {output}")

    for W in weight_list:
        print(f"computing dot of {output} with {W}")
        output = W.dot(output)
        print(f"output: {output}")

    return output


main()
