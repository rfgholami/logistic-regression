import numpy as np

from sigmoid import sigmoid


def compute_cost(X, y, theta):
    epsilon = 1e-5
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))
    # cost = (1. / m) * np.sum(np.multiply(-y, np.log(h)) - np.multiply((1. - y), np.log(1. - h)), axis=0)
    cost = (-y * np.log(h+epsilon) - (1 - y) * np.log(1 - h+epsilon)).mean()

    return cost


