import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_prob(self, x):
        return self.sigmoid(np.dot(x, self.w) + self.b)

    def predict(self, X):
        prob = self.predict_prob(X)
        labels = (prob >= 0.5).astype(int)
        return prob, labels

    def loss_function(self, X, y):
        m = X.shape[0]
        z = np.dot(X, self.w) + self.b
        loss = -y * z + np.log(1 + np.exp(z))
        return np.mean(loss)

    def calculate_grad(self, X, y):
        m = X.shape[0]
        prob = self.sigmoid(np.dot(X, self.w) + self.b)
        grad_w = (1 / m) * np.dot(X.T, (prob - y))
        grad_b = (1 / m) * np.sum(prob - y)
        return grad_w, grad_b

    def gradient_descent(self, X, y, learning_rate, max_iter, epsilon):
        loss_list = []
        for i in range(max_iter):
            loss_old = self.loss_function(X, y)
            loss_list.append(loss_old)
            grad_w, grad_b = self.calculate_grad(X, y)
            self.w -= learning_rate * grad_w
            self.b -= learning_rate * grad_b
            loss_new = self.loss_function(X, y)
            if np.abs(loss_new - loss_old) <= epsilon:
                break
        return loss_list

    def fit(self, X, y, learning_rate=0.01, max_iter=1000, epsilon=1e-5):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0
        loss_list = self.gradient_descent(X, y, learning_rate, max_iter, epsilon)
        return loss_list

# Generate data
data, labels = make_blobs(n_samples=200, n_features=2, centers=2)

# Initialize logistic regression model
lr = LogisticRegression()

# Train the model
loss_list = lr.fit(data, labels)

# Plot the data points and decision boundary
plt.scatter(data[labels == 1, 0], data[labels == 1, 1], s=30, c="b", marker="o", label="class 1")
plt.scatter(data[labels == 0, 0], data[labels == 0, 1], s=30, c="r", marker="x", label="class 0")
plt.xlabel("x1")
plt.ylabel("x2")

x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
x_range = np.linspace(x_min, x_max, 200)
y_range = (-lr.b - lr.w[0] * x_range) / lr.w[1]
plt.plot(x_range, y_range, color="red")

plt.legend()
plt.savefig("result.png", bbox_inches='tight', dpi=400)
plt.show()