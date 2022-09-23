

import numpy as np

from TrainingTask import TrainingTask
from utils import algorithm
from utils.network import Network

np.random.seed(1234)

# This code goes into jupyter notebook


def main():
    dx = 2  # total features
    dz = 1  # output vector size
    di = 2  # neurons in each layer
    H = 1  # number of hidden layers

    input_data = np.array([0.53436063, 0.31448347])
    z = np.array([[0.18450251],
                  [0.28731715]])
    network = Network(dx, dz, di, H)

    task = TrainingTask(network=network, input_data=input_data, z=z)
    task.initialize()
    output = task.start_training(algorithm_function=algorithm.gradient_descent)


main()
