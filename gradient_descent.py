import numpy as np

from sigmoid import sigmoid


def gradient_descent(X, y, theta, learning_rate, m):
    h = sigmoid(np.dot(X, theta))
    # grad = (np.dot(X.T, (h - y)))
    grad = np.sum(np.multiply(X, (h - y)),axis=0)


    grad = np.multiply((learning_rate / m), grad)
    return grad.reshape((3, 1))
