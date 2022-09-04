import numpy as np
from numpy import genfromtxt


def get_random_data(rows, columns):
    return np.random.randint(5, size=(rows, columns))


def load_csv(file):
    my_data = genfromtxt(file, delimiter=',')
    return my_data
