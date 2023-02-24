import logging
import numpy as np

from verifiNN.trainer import TrainingTask
from verifiNN.models.linear_regression import LinearRegression
from verifiNN.algorithms.algorithm import gradient_descent


def main():
	logging.basicConfig(level=logging.INFO)

	X = np.array([[1, 2], [3, 4]])
	Z = np.array([5, 6])
	p = np.array([1, 1, 1])

	lr = LinearRegression()
	lr.initialize(params=p)

	tt = TrainingTask(lr, X, Z, gradient_descent)
	tt.start()


if __name__ == "__main__":
	main()