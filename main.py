from compute_cost import compute_cost
from gradient_descent import gradient_descent
from predict import predict
from utils import load_dataset, add_x0, feature_normalize
import numpy as np
import matplotlib.pyplot as plt

data = load_dataset("data.txt")

X_Original = data[:, 0:2]
y = data[:, 2:3]

plt.scatter(X_Original[:, 0], X_Original[:, 1], c=y, s=50, cmap=plt.cm.Spectral)

X, mu, sigma = feature_normalize(X_Original)

plt.show()

X = add_x0(X)
m = X.shape[0]
n = X.shape[1]
learning_rate = .3
theta = np.zeros((n, 1))
max_iter = 400

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

best_theta = np.array([[-25.161272],
                       [0.206233],
                       [0.201470]], dtype=float)

xa1 = 0
xb1 = -(best_theta[0] + xa1 * best_theta[1]) / best_theta[2]
xb2 = 0
xa2 = -(best_theta[0] + xb2 * best_theta[2]) / best_theta[1]

xa1 = (xa1 - mu[0]) / sigma[0]
xb1 = (xb1 - mu[1]) / sigma[1]
xa2 = (xa2 - mu[0]) / sigma[0]
xb2 = (xb2 - mu[1]) / sigma[1]

plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=1, color='blue', label='best_theta')

xa1 = -3
xb1 = -(theta[0] + xa1 * theta[1]) / theta[2]

xb2 = -3
xa2 = -(theta[0] + xb2 * theta[2]) / theta[1]

plt.plot([xa1, xa2], [xb1, xb2], 'k-', lw=1, color='red', label='my_theta')
plt.legend(loc='upper center', shadow=True)

plt.show()

x = np.array([
    [80.27957401466998, 92.11606081344084],
    [75.47770200533905, 90.42453899753964],
    [78.63542434898018, 96.64742716885644],
    [52.34800398794107, 60.76950525602592],
    [94.09433112516793, 77.15910509073893],
    [90.44855097096364, 87.50879176484702],
    [55.48216114069585, 35.57070347228866],
    [74.49269241843041, 84.84513684930135],
    [89.84580670720979, 45.35828361091658],
    [83.48916274498238, 48.38028579728175],
    [42.2617008099817, 87.10385094025457],
    [99.31500880510394, 68.77540947206617],
    [55.34001756003703, 64.9319380069486],
    [74.77589300092767, 89.52981289513276],
    [10.77589300092767, 10.52981289513276],
    [100.77589300092767, 100.52981289513276],
    [-0.77589300092767, -0.52981289513276],
], dtype=float)

predicted1 = np.array(predict(x, best_theta, 0, 1) > .5, dtype=int)
predicted2 = np.array(predict(x, theta, mu, sigma) > .5, dtype=int)

print("predicted value with best_theta = " + str(predicted1))
print("predicted value with theta = " + str(predicted2))
