# import necessary lib
import numpy as np
import matplotlib.pyplot as plt


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


X = np.linspace(-1, 1, 200)
Y = 2 * X + np.random.randn(200) * 0.3
train_x = X.reshape(-1, 1)
train_y = Y.reshape(-1, 1)
LR = LinearRegression()
LR.fit(train_x, train_y, 0.01, 1000, 0.00001)
plt.plot(X, Y, 'ro', label="trainning data")
plt.legend()
plt.plot(X, LR.w[0, 0] * X + LR.b, ls="-", lw=2, c="b")
plt.xlabel("x")
plt.ylabel("y")
s = "y=%.3f*x%.3f" % (LR.w[0, 0], LR.b)
plt.text(0, LR.b - 0.2, s, color="b")
plt.savefig("result.png", bbox_inches='tight', dpi=400)
plt.show()
