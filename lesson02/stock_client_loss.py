# -*- coding: GB2312 -*-

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

f = open('Stock_Client_Loss.csv', encoding='GB2312')
data = pd.read_csv(f)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, :4], data.iloc
[:, 5], test_size=0.3, shuffle=True)


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

    def fit(self, X, y, learning_rate=1e-10, max_iter=500, epsilon=1e-5):
        n_features = X.shape[1]
        self.w = np.zeros(n_features)
        self.b = 0.0
        loss_list = self.gradient_descent(X, y, learning_rate, max_iter, epsilon)
        return loss_list


LR = LogisticRegression()
train_x = np.array(X_train).reshape(-1, 4)
train_y = np.array(Y_train).reshape(-1, 1)
test_x = np.array(X_test).reshape(-1, 4)
test_y = np.array(Y_test).reshape(-1, 1)
train_yy=train_y.flatten()
LR.fit(train_x, train_yy)
# 进行预测
predicted_y = LR.predict_prob(test_x)
# 评估模型的预测性能
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("均方误差 (MSE):", mse)
print("确定系数 (R-squared):", r2)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
train_x = np.array(X_train).reshape(-1, 4)
train_y = np.array(Y_train).reshape(-1, 1)
test_x = np.array(X_test).reshape(-1, 4)
test_y = np.array(Y_test).reshape(-1, 1)
train_yy=train_y.flatten()
LR.fit(train_x, train_yy)
# 进行预测
predicted_y = LR.predict(test_x)
# 评估模型的预测性能
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("均方误差 (MSE):", mse)
print("确定系数 (R-squared):", r2)
