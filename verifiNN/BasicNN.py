import numpy as np

layers = 2  # total layers
neuron_layer = [1, 1]  # neurons per layer


def main():
    print("starting basic neural network training")
    print(f"layers: {layers}")
    print(f"neuron per layer: {neuron_layer}")

    output = compute_nn_output_one_iteration()
    print(f"output after 1 iteration: {output}")


def compute_nn_output_one_iteration():
    input_data = get_training_data()
    for layer in neuron_layer:
        input_data = compute_layer_output(layer, input_data)

    return input_data


def get_training_data():
    return np.array([5, 10])


# returns vector with dimension = neuron_count
def compute_layer_output(neuron_count, input):
    layer_list = []

    for i in range(neuron_count):
        layer_output = compute_neuron_output(input) # pass a weights vector
        layer_list.append(layer_output)

    return np.array(layer_list)


def compute_neuron_output(input):
    # TODO  have to call this only once
    weights = initialize_weights(input.shape[0])
    print(f"weights: {weights}")
    print(f"input_data: {input}")
    weighted_sum = compute_weighted_sum(weights, input)
    return compute_activation(weighted_sum)


# returns vector with d = feature_count (incoming data to a layer)
def initialize_weights(feature_count):
    return np.ones(feature_count)


def compute_weighted_sum(weights, data):
    return weights.transpose().dot(data)


def compute_activation(input):
    print("activation:to be implemented")
    return input


def compute_error():
    print("to be implemented")


main()
