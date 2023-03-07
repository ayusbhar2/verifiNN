import logging
import numpy as np

from verifiNN.trainer import TrainingTask
from verifiNN.models.models import LinearRegression, LogisticRegression
from verifiNN.algorithms.algorithm import gradient_descent


def main():
	logging.basicConfig(level=logging.INFO)

	# ~ Linear Regression ~ #

	X = np.array([[1, 2], [3, 4]])
	Z = np.array([5, 6])
	p = np.array([1, 1, 1])

	lr = LinearRegression()
	lr.initialize(params=p)

	tt = TrainingTask(lr, X, Z, gradient_descent, max_iters=10)
	tt.start()

	# ~ Logistic Regression ~ #

	X = np.array([[1, 2], [3, 4]])
	Z = np.array([0, 1])
	p = np.array([1, 1, 1])

	lor = LogisticRegression()
	lor.initialize(params=p)

	tt1 = TrainingTask(lor, X, Z, gradient_descent, max_iters=10)
	tt1.start()


if __name__ == "__main__":
	main()