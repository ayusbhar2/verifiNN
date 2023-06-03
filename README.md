# verifiNN

*Robustness* is a desirable property in a neural network. Informally, robustness can be described as ‘resilience to perturbations in the input’. Said differently, a neural network is robust if small changes to the input produce small or no changes to the output. In particular, if the network is a classifier, robustness means that inputs close to each other should be assigned the same class by the network.

This project implements convex optimization based methods for robustness verification of neural networks. Given a trained neural network and an input, we use an optimization based approach to determine if the network is robust at the input point. Currently only a Linear Programming based approach is supported for ReLU as well as Idenditity activated feed-forward neural networks. Future work will include a Semidefinite Programming based approach for fully connected as well as convolutional neural networks.

For a detailed treatment of the mathematical background, check out this [blog post](https://ayusbhar2.github.io/verifying-neural-nertwork-robustness-using-linear-programming/). Here is a small example on how to use `verifiNN`.

## Example

```{python}
pip install verifiNN
```

```{python}
import numpy as np

from verifiNN.models.network import Network
from verifiNN.verifier import LPVerifier
```

Here we generate a toy network for our example. In reality, this network would be given to us.

```{python}
# Defining a network
W1 = np.array([[1, 0],
              [0, 1]])
b1 = np.array([1, 1])
W2 = np.array([[0, 1],
              [1, 0]])
b2 = np.array([2, 2])

weights = [W1, W2]
biases = [b1, b2]
network = Network(weights, biases, activation='ReLU', labeler='argmax')
```

Next, we note the class label that the network assigns to a reference input `x_0`.

```{python}
x_0 = np.array([1, 2])
l_0 = network.classify(x_0)  # class 0
assert l_0 == 0
```

Then, we compute the *pointwise robustness* (i.e. the distance to the nearest adversarial example within an $\epsilon-$Ball around the reference point.

```{python}
epsilon = 1.5

vf = LPVerifier()
result = vf.compute_pointwise_robustness(network, x_0, epsilon)
assert result['verification_status'] == 'verified'
assert result['robustness_status'] == 'not_robust'
```

`verifiNN` was able to verify that the above nework is NOT robust at `x_0`. This is because an adversarial example was found within the $\epsilon-$Ball around `x_0` (as shown below).

```
rho = np.round(result['pointwise_robustness'], decimals=5)
assert rho == 0.5  # distanc to nearest adverarial example

x_hat = result['adversarial_example']
assert np.round(x_hat[0], decimals=5) == 1.5
assert np.round(x_hat[1], decimals=5) == 1.5

assert network.classify(x_hat) == 1  # class 1
```

The adversarial example `(1.5, 1.5)` lies inside (actually, on the boundary of) the $\epsilon-$Ball around `x_0`. Yet, as expected, the network assigns the class label `1` to `x_hat`.

**Caution**: `verifiNN` currently suffers from a limitation - if an adversarial example is found, then clearly the network is not robust. However, the converse is not true. In other words, if no adversarial example was found (i.e. the underlyin optimization problem was infeasible) we cannot conclude that the network is robust. This limitation comes from the affine appoximation of the ReLU function in the current lineaer programming based approach. Alternative appraoches (to be implemented in the future) do not suffer from this limitation.


## References:
- [Verifying Neural Network Robustness with Lineear Programming](https://ayusbhar2.github.io/verifying-neural-nertwork-robustness-using-linear-programming/)