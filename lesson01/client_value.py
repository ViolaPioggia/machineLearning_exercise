import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

f = open('Client_Value.csv')
data = pd.read_csv(f)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:, 1:], data.iloc[:, 0], test_size=0.3, shuffle=True)


# Add code here to build and test LinearRegression Model
# define LinearRegression class
class LinearRegression:
    # define prediction function
    def __init__(self):
        self.w = 0
        self.b = 0

    def predict(self, x):
        """
        Parameters
        ----------
        x : TYPE:array
        DESCRIPTION: feature vector of instance
        Returns
        -------
        y:predicted label value
        """
        # Add code here to calculate prediction, see Eq.(1.6)
        y = x.dot(self.w) + self.b
        return y

    # define loss function
    def loss_function(self, train_x, train_y):
        """
        Parameters
        ----------
        train_x : TYPE: (m,d) 2-D numpy array
        DESCRIPTION: feature space, where each row represents
        a fature vector of a training instance
        1.3 实验内容 –5/85–
        train_y : TYPE:numpy array with m elements
        DESCRIPTION: each element represents the label value of
        a training instance
        Returns
        -------
        loss: loss function values.
        """

        inst_num = train_x.shape[0]  # data size
        pred_y = train_x.dot(self.w) + self.b  # training prediction
        loss = np.sum((pred_y - train_y) ** 2) / (2 * inst_num)  # see Eq.(1.7)
        return loss

    # define gradient calculation function
    def calculate_grad(self, train_x, train_y):
        """
        Parameters
        ----------
        train_x : TYPE: (m,d) 2-D numpy array
        DESCRIPTION: feature space, where each row represents
        a fature vector of a training instance
        train_y : TYPE:numpy array with m elements
        DESCRIPTION: each element represents the label value of
        a training instance
        Returns
        -------
        grad_w: gradients for cofficients w.
        grad_b: gradient for bias b
        """

        inst_num = train_x.shape[0]  # data size
        pred_y = train_x.dot(self.w) + self.b  # training prediction
        # Add code here to calculate grad of weights, see Eq.(1.8)
        grad_w = 1 / inst_num * train_x.T.dot(pred_y - train_y)
        # Add code here to calculate grad of bias, see Eq.(1.9)
        grad_b = 1 / inst_num * sum(pred_y - train_y)
        return grad_w, grad_b

    # gradient descent algorithm
    def gradient_descent(self, train_x, train_y, learn_rate, max_iter, epsilon):
        """
        Parameters
        ----------
        train_x : TYPE: (m,d) 2-D numpy array
        DESCRIPTION: feature space, where each row represents
        a fature vector of a training instance
        train_y : TYPE:numpy array with m elements
        DESCRIPTION: each element represents the label value of
        a training instance
        1.3 实验内容 –6/85–
        learn_rate:TYPE: float
        max_iter: TYPE: int
        DESCRIPTION: max iterations
        epsilon: TYPE: float
        DESCRIPTION: permitted errors between two iterations
        Returns
        -------
        loss_list: TYPE: list
        DESCRIPTION: loss value for each iteration
        """

        loss_list = []
        for i in range(max_iter):
            loss_old = self.loss_function(train_x, train_y)
            loss_list.append(loss_old)
            grad_w, grad_b = self.calculate_grad(train_x, train_y)
            self.w = self.w - learn_rate * grad_w
            self.b = self.b - learn_rate * grad_b
            loss_new = self.loss_function(train_x, train_y)
            if abs(loss_new - loss_old) <= epsilon:
                break
        return loss_list

    # learning linear regression model
    def fit(self, train_x, train_y, learn_rate, max_iter, epsilon):
        """
        Parameters
        ----------
        train_x : TYPE: (m,d) 2-D numpy array
        DESCRIPTION: feature space, where each row represents
        a fature vector of a training instance
        train_y : TYPE:numpy array with m elements
        DESCRIPTION: each element represents the label value of
        a training instance
        learn_rate:TYPE: float
        max_iter: TYPE: int
        DESCRIPTION: max iterations
        epsilon: TYPE: float
        DESCRIPTION: permitted errors between two iterations
        Returns
        -------
        loss_list: TYPE: list
        DESCRIPTION: loss value for each iteration
        """

        feat_num = train_x.shape[1]  # feature dimension
        self.w = np.zeros((feat_num, 1))  # initialize model parameters
        self.b = 0.0
        # learn model parameters using gradient descent algorithm
        loss_list = self.gradient_descent(train_x, train_y, learn_rate, max_iter
                                          , epsilon)
        self.training_visualization(loss_list)

    # learning process visualization
    def training_visualization(self, loss_list):
        """
        Parameters
        ----------
        loss_list: TYPE: list
        DESCRIPTION: loss value for each iteration
        """

        plt.plot(loss_list, color='red')
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.show()
        print(loss_list)


LR = LinearRegression()
# Add code here to build and test LinearRegression Model
train_x = np.array(X_train).reshape(-1, 5)
train_y = np.array(Y_train).reshape(-1, 1)
test_x = np.array(X_test).reshape(-1, 5)
test_y = np.array(Y_test).reshape(-1, 1)
LR.fit(train_x, train_y, 1e-10, 500, 0.00001)

# 进行预测
predicted_y = LR.predict(test_x)

# 评估模型的预测性能
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("均方误差 (MSE):", mse)
print("确定系数 (R-squared):", r2)

from sklearn.linear_model import LinearRegression

LR = LinearRegression()
# Add code here to build and test LinearRegression Model
train_x = np.array(X_train).reshape(-1, 5)
train_y = np.array(Y_train).reshape(-1, 1)
test_x = np.array(X_test).reshape(-1, 5)
test_y = np.array(Y_test).reshape(-1, 1)
LR.fit(train_x, train_y)

# 进行预测
predicted_y = LR.predict(test_x)

# 评估模型的预测性能
mse = mean_squared_error(test_y, predicted_y)
r2 = r2_score(test_y, predicted_y)

print("均方误差 (MSE):", mse)
print("确定系数 (R-squared):", r2)