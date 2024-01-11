import struct
import os

import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Parameters
        ----------
        layer_sizes : TYPE:list
        DESCRIPTION: store node numbers of each layer
        Returns
        -------
        None.
        """
        self.num_layers = len(layer_sizes)  # layer number of NN
        self.layers = layer_sizes  # node numbers of each layer
        # initialize connenct weights of layers
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1],
                                                              layer_sizes[1:])]
        # initialize biases of each layer(input layer has no bias)
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    # sigmoid activation function
    def sigmoid(self, x):
        y = x.copy()  # 对sigmoid函数优化，避免出现极大的数据溢出
        y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
        y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        return y

    # derivative function of sigmoid activation function
    def sigmoid_prime(self, z):
        """
        Parameters
        ----------
        z : input of activation derivative function
        Returns
        4.3 实验内容 –35/85–
        -------
        act: derivative activation function value
        """
        act = self.sigmoid(z) * (1.0 - self.sigmoid(z))
        return act

    # feed forward to get prediction
    def feed_forward(self, x):
        """
        Parameters
        ----------
        x : 2-D array, matrix of feature vectors of training instance
        Returns
        -------
        output: results of output layer
        """
        output = x.copy()
        for w, b in zip(self.weights, self.biases):
            output = self.sigmoid(np.dot(w, output) + b)
        return output

    # feed backward to update NN paremeters
    def feed_backward(self, x, y):
        """
        Parameters
        ----------
        x : 2-D array, matrix of feature vectors of training instances
        y : 2-D array, maxtrix of label vectors of training instances
        Returns
        -------
        delta_w: update of weights
        delta_b: update of biases
        """
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        # activations of input layer
        activation = np.transpose(x)
        activations = [activation]
        # input after input layer
        layer_input = []
        # forward to get each layer’s input and output
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            layer_input.append(z)  # input of each layer
            activation = self.sigmoid(z)
            activations.append(activation)  # output of each layer
        # loss funtion
        ground_truth = np.transpose(y)
        diff = activations[-1] - ground_truth
        # get input of last layer
        last_layer_input = layer_input[-1]
        delta = np.multiply(diff, self.sigmoid_prime(last_layer_input))
        # bias update of last layer
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        # weight update of last layer
        delta_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # update weights and bias from 2nd layer to last layer
        for i in range(2, self.num_layers):
            input_values = layer_input[-i]
            delta = np.multiply(np.dot(np.transpose(self.weights[-i + 1]), delta), \
                                self.sigmoid_prime(input_values))
            delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
            delta_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
        return delta_b, delta_w

    # training using BP
    def fit(self, x, y, learnrate, mini_batch_size, epochs=1000):
        """
        Parameters
        ----------
        x : 2-D array of training feature vectors
        y : 2-D array of training label vectors.
        learnrate : float, learn rate.
        mini_batch_size : int,batch size
        epochs : int, optional, The default is 1000.
        Returns
        -------
        None.
        """
        n = len(x)  # training size
        for i in range(epochs):
            randomlist = np.random.randint(0, n - mini_batch_size, int(n / mini_batch_size))
            batch_x = [x[k:k + mini_batch_size] for k in randomlist]
            batch_y = [y[k:k + mini_batch_size] for k in randomlist]
            for j in range(len(batch_x)):
                delta_b, delta_w = self.feed_backward(batch_x[j], batch_y[j])
                self.weights = [w - (learnrate / mini_batch_size) * dw for w, dw in
                                zip(self.weights, delta_w)]
                self.biases = [b - (learnrate / mini_batch_size) * db for b, db in
                               zip(self.biases, delta_b)]
            if (i + 1) % 100 == 0:
                labels = self.predict(x)
                acc = 0.0
                for k in range(len(labels)):
                    if y[k, labels[k]] == 1.0:
                        acc += 1.0
                acc = acc / len(labels)
                print("iterations %d accuracy %.3f" % (i + 1, acc))

    # predict function
    def predict(self, x):
        """
        Parameters
        ----------
        x : 2-D array of feature vectors of test instances
        Returns
        -------
        labels : predicted labels.
        """
        results = self.feed_forward(x.T)
        labels = [np.argmax(results[:, y]) for y in range(results.shape[1])]
        return labels


# fucntion to load MNIST data
def load_mnist_data(path, kind='train'):
    """
    Parameters
    ----------
    path : str, file path
    kind : str, optional,The default is ’train’.
    to demostrate load train or test data
    Returns
    -------
    images : 2-D array, feature vectors of images
    labels : list, corresponding labels of images.
    """
    label_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(label_path, 'rb') as lbpath:  # open label file
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(image_path, 'rb') as imgpath:  # open image file
        struct.unpack('>IIII', imgpath.read(16))
        # transform image into 784-dimensional feature vector
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


import matplotlib.pyplot as plt


def show_image(image):
    plt.figure()
    img = image.reshape(28, 28)
    plt.imshow(img, 'gray')
    plt.show()


from sklearn.preprocessing import StandardScaler

path = './'
train_images, train_labels = load_mnist_data(path, kind='train')
train_y = np.zeros((len(train_labels), 10))
for i in range(len(train_labels)):
    train_y[i, train_labels[i]] = 1
scaler = StandardScaler()
train_x = scaler.fit_transform(train_images)
test_images, test_labels = load_mnist_data(path, kind='t10k')
test_y = np.zeros((len(test_labels), 10))
for i in range(len(test_labels)):
    test_y[i, test_labels[i]] = 1
test_x = scaler.fit_transform(test_images)

layer_sizes = [784, 100, 10]
NN = NeuralNetwork(layer_sizes)
NN.fit(train_x, train_y, learnrate=0.01, mini_batch_size=100, epochs=500)

test_pred_labels = NN.predict(test_x)
acc = 0.0
for k in range(len(test_pred_labels)):
    if test_y[k, test_pred_labels[k]] == 1.0:
        acc += 1.0
acc = acc / len(test_pred_labels)
print("test accuracy:%.3f" % (acc))

from sklearn.neural_network._multilayer_perceptron import MLPClassifier

model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic',
                      solver='sgd', batch_size=100, learning_rate='constant',
                      learning_rate_init=0.01, max_iter=3000)
model.fit(train_x, train_y)
labels = model.predict(test_x)
acc = 0.0
for k in range(len(labels)):
    index = 0
    for j in range(10):
        if labels[k, j] == 1:
            index = j
            break
    if test_y[k, index] == 1.0:
        acc += 1.0
acc = acc / len(labels)
print("test accuracy:%.3f" % (acc))
