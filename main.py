from compute_cost import compute_cost
from gradient_descent import gradient_descent
from utils import load_dataset, add_x0
import numpy as np
import matplotlib.pyplot as plt

data = load_dataset()

X_Original = data[:, 0:2]
y = data[:, 2:3]

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

plt.show()

X = add_x0(X_Original)
m = X.shape[0]
n = X.shape[1]
learning_rate = .0002
theta = np.zeros((n, 1))

his = np.zeros((2000000, 1))

for i in range(2000000):
    cost = compute_cost(X, y, theta)
    grad = gradient_descent(X, y, theta, learning_rate, m)
    theta = theta - grad

    his[i, :] = cost

plt.plot(his, label='cost')

plt.ylabel('cost')
plt.xlabel('step')
plt.title("linear regression'")

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()

plt.show()

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

theta2 = np.array([[-25.161272],
                   [0.206233],
                   [0.201470]], dtype=float)


xa1 = 0
xb1 = -(theta2[0] + xa1 * theta2[1]) / theta2[2]
xb2 = 0
xa2 = -(theta2[0] + xb2 * theta2[2]) / theta2[1]
plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=2)

xa1 = 0
xb1 = -(theta[0] + xa1 * theta[1]) / theta[2]
xb2 = 0
xa2 = -(theta[0] + xb2 * theta[2]) / theta[1]
plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=2)



plt.show()
