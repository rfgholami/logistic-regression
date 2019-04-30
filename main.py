from compute_cost import compute_cost
from gradient_descent import gradient_descent
from utils import load_dataset, add_x0, feature_normalize
import numpy as np
import matplotlib.pyplot as plt

data = load_dataset()

X_Original = data[:, 0:2]
y = data[:, 2:3]

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

X, mu, sigma = feature_normalize(X_Original)

plt.show()

X = add_x0(X)
m = X.shape[0]
n = X.shape[1]
learning_rate = .01
theta = np.zeros((n, 1))

his = np.zeros((1000, 1))

for i in range(1000):

    cost = compute_cost(X, y, theta)
    grad = gradient_descent(X, y, theta, learning_rate, m)
    theta = theta - grad

    his[i, :] = cost

    # print (add_to_learning_rate)
    if i % 100 == 0:
        print (str(i) + "--" + str(cost))

plt.plot(his, label='cost')

plt.ylabel('cost')
plt.xlabel('step')
plt.title("logistic regression'")

plt.legend(loc='upper center', shadow=True)

plt.show()

plt.scatter(X[:, 1], X[:, 2], c=y, s=50, cmap=plt.cm.Spectral)

old_theta = np.array([[-25.161272],
                      [0.206233],
                      [0.201470]], dtype=float)

xa1 = 0
xb1 = -(old_theta[0] + xa1 * old_theta[1]) / old_theta[2]
xb2 = 0
xa2 = -(old_theta[0] + xb2 * old_theta[2]) / old_theta[1]

xa1 = (xa1 - mu) / sigma
xb1 = (xb1 - mu) / sigma
xa2 = (xa2 - mu) / sigma
xb2 = (xb2 - mu) / sigma

plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=1, color='blue', label='old_theta')

xa1 = 0
xb1 = -(theta[0] + xa1 * theta[1]) / theta[2]

xb1 = -(xb1 - mu) / sigma - theta[0]
xa1 = (xa1 - mu) / sigma

xb2 = 0
xa2 = -(theta[0] + xb2 * theta[2]) / theta[1]

xa2 = -(xa2 - mu) / sigma - theta[0]
xb2 = (xb2 - mu) / sigma

plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=1, color='red', label='theta')
plt.legend(loc='upper center', shadow=True)

plt.show()
