import numpy as np


def sigmoid(z):
    g = 1. / (np.exp(-z) + 1.)
    return g
