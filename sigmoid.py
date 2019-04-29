import numpy as np
import math


def sigmoid(z):
    g = 1. / (np.exp(-z) + 1.)
    return g
