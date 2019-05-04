from compute_cost import compute_cost
from gradient_descent import gradient_descent
from predict import predict
from sigmoid import sigmoid
from utils import load_dataset, add_x0, feature_normalize, get_arange, \
    add_polynomial_feature, multiply_feature, remove_out_of_bound_values
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

data = load_dataset("data.txt")

X_Original = data[:, 0:2]
y = data[:, 2:3]

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

X_Original = add_polynomial_feature(X_Original, np.array([0, 0]))
X_Original = add_polynomial_feature(X_Original, np.array([0, 1]))
X_Original = add_polynomial_feature(X_Original, np.array([1, 1]))

X, mu, sigma = feature_normalize(X_Original)

plt.show()

X = add_x0(X)
m = X.shape[0]
n = X.shape[1]
learning_rate = .05
theta = np.zeros((n, 1))
max_iter = 30000

his = np.zeros((max_iter, 1))

for i in range(max_iter):

    cost = compute_cost(X, y, theta)
    grad = gradient_descent(X, y, theta, learning_rate, m)
    theta = theta - grad

    his[i, :] = cost

    if i % 100 == 99:
        print ("iterate number: " + str(i + 1) + " -- cost: " + str(cost))

plt.plot(his, label='cost')

plt.ylabel('cost')
plt.xlabel('step')
plt.title("logistic regression'")

plt.legend(loc='upper center', shadow=True)

plt.show()

# plt.scatter(X[:, 1], X[:, 2], c=y, s=50, cmap=plt.cm.Spectral)

xa = get_arange(X[:, 1])
xb = get_arange(X[:, 2])

yhat = np.zeros((m, m))

for i in range(xa.shape[0]):
    yhat[i, :] = theta[0] + theta[1] * xa[i] + theta[2] * xb + theta[3] * (np.multiply(xa[i], xa[i])) + theta[4] * (
        np.multiply(xa[i], xb)) + \
                 theta[5] * (np.multiply(xb, xb))



yhat = sigmoid(yhat)

plt.scatter(X[:, 1], X[:, 2], c=y, s=50, cmap=plt.cm.Spectral)
plt.contour(xa, xb, yhat, 50, cmap='binary')

plt.show()





for i in range(xa.shape[0]):
    xaT = np.ones(xa.shape) * xa[i]

    yhat2 = theta[0] + theta[1] * xaT + theta[2] * xb + theta[3] * (np.multiply(xaT, xaT)) + theta[4] * (
        np.multiply(xaT, xb)) + \
            theta[5] * (np.multiply(xb, xb))
    yhat2 = sigmoid(yhat2)
    yhat2 = yhat2 > .5

    plt.scatter(xaT, xb, c=yhat2, s=50, cmap=plt.cm.Spectral)
plt.show()
