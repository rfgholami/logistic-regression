from compute_cost import compute_cost
from gradient_descent import gradient_descent
from predict import predict
from utils import load_dataset, add_x0, feature_normalize, load_dataset2
import numpy as np
import matplotlib.pyplot as plt

data = load_dataset("data2.txt")

X_Original = data[:, 0:3]
y = data[:, 3:4]

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

X, mu, sigma = feature_normalize(X_Original)

plt.show()

X = add_x0(X)
m = X.shape[0]
n = X.shape[1]
learning_rate = .3
theta = np.zeros((n, 1))
max_iter = 8000

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




plt.scatter(X[:, 1], X[:, 2], c=y, s=50, cmap=plt.cm.Spectral)
xa1 = -2
xc1 = 0
xb1 = -(theta[0] + xa1 * theta[1] + xc1 * theta[3]) / theta[2]
xb2 = -2
xc2 = 0
xa2 = -(theta[0] + xb2 * theta[2] + xc2 * theta[3]) / theta[1]
plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=1, color='red', label='my_theta')
plt.legend(loc='upper center', shadow=True)
plt.show()




plt.scatter(X[:, 1], X[:, 3], c=y, s=50, cmap=plt.cm.Spectral)
xa1 = -2
xb1 = 0
xc1 = -(theta[0] + xa1 * theta[1] + xb1 * theta[2]) / theta[3]

xc2 = -2
xb2 = 0
xa2 = -(theta[0] + xb2 * theta[2] + xc2 * theta[3]) / theta[1]
plt.plot([xa1, xa2], [xc1, xc2], 'k-', lw=1, color='red', label='my_theta')
plt.legend(loc='upper center', shadow=True)
plt.show()




plt.scatter(X[:, 2], X[:, 3], c=y, s=50, cmap=plt.cm.Spectral)
xa1 = 0
xc1 = -2
xb1 = -(theta[0] + xa1 * theta[1] + xc1 * theta[3]) / theta[2]
xb2 = -2
xa2 = 0
xc2 = -(theta[0] + xb2 * theta[2] + xa2 * theta[1]) / theta[3]
plt.plot([xc1, xc2], [xb1, xb2], 'k-', lw=1, color='red', label='my_theta')
plt.legend(loc='upper center', shadow=True)
plt.show()




