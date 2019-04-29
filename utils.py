import numpy as np


def load_dataset():
    f = open("data.txt", "r")
    contents = f.read()

    lines = contents.split("\n")

    data = np.ndarray(shape=(lines.__len__(), 3), dtype=float, order='F')

    for i in range(lines.__len__()):
        items = lines[i].split(",")
        items = [float(j) for j in items]
        data[i, 0] = items[0]
        data[i, 1] = items[1]
        data[i, 2] = items[2]
    return data


def feature_normalize(X):
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def add_x0(X_Original):
    m = X_Original.shape[0]
    n = X_Original.shape[1]
    X = np.ones((m, (n + 1)), dtype=float)
    X[:, 1:(n + 1)] = X_Original
    return X
