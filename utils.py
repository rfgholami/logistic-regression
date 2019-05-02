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
    mu = np.mean(X, axis=0)

    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


def add_x0(X_Original):
    m = X_Original.shape[0]
    n = X_Original.shape[1]
    X = np.ones((m, (n + 1)), dtype=float)
    X[:, 1:(n + 1)] = X_Original
    return X


def add_polynomial_feature(X_Original, ls):
    val = X_Original[:, ls[0]]

    for i in range(ls.shape[0] - 1):
        nx = X_Original[:, ls[i + 1]]
        val = np.multiply(val, nx)

    m = X_Original.shape[0]
    n = X_Original.shape[1]
    X = np.zeros((m, (n + 1)), dtype=float)
    X[:, 0:n] = X_Original
    X[:, n] = val
    return X


def multiply_feature(x, ls):
    val = x

    for i in range(ls.shape[0] - 1):
        nx = x
        val = np.multiply(val, nx)

    return val


def get_arange(X):
    minVal = np.min(X)
    maxVal = np.max(X)
    num = (maxVal - minVal) / 100
    return np.arange(minVal, maxVal, num)


def remove_out_of_bound_values(X, xb):
    xb_minVal = np.min(X[:, 2])
    xb_maxVal = np.max(X[:, 2])

    xb = xb * (xb > xb_minVal)
    xb_min = xb_minVal * (xb == 0)
    xb = xb + xb_min

    xb = xb * (xb < xb_maxVal)
    xb_max = xb_maxVal * (xb == 0)
    xb = xb + xb_max

    return xb
