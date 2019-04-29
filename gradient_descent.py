import numpy as np


def gradient_descent(X, y, theta, learning_rate, m):
    h = np.dot(X, theta)
    grad = (np.dot(X.T, (h - y)))
    grad = np.multiply((learning_rate / m), grad)
    return grad
