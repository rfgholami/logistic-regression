import numpy as np


def load_dataset(file_name):
    f = open(file_name, "r")
    contents = f.read()

    lines = contents.split("\n")

    data = np.ndarray(shape=(lines.__len__(), lines[0].__len__()), dtype=float, order='F')

    for i in range(lines.__len__()):
        items = lines[i].split(",")
        items = [float(j) for j in items]
        for j in range(items.__len__()):
            data[i, j] = items[j]
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
