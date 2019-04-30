import numpy as np

from sigmoid import sigmoid
from utils import add_x0


def predict(x_orginal, theta, mu, sigma):
    x_orginal = (x_orginal - mu) / sigma

    x = add_x0(x_orginal)

    return sigmoid(np.dot(x, theta))
