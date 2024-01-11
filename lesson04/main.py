import numpy as np
class NeuralNetwork:
    def __init__(self,layer_sizes):
        """
        Parameters
        ----------
        layer_sizes : TYPE:list
        DESCRIPTION: store node numbers of each layer
        Returns
        -------
        None.
        """
        self.num_layers = len(layer_sizes) # layer number of NN
        self.layers = layer_sizes # node numbers of each layer
        #initialize connenct weights of layers
        self.weights = [np.random.randn(y,x) for x, y in zip(layer_sizes[:-1],
        layer_sizes[1:])]
        #initialize biases of each layer(input layer has no bias)
        self.biases = [np.random.randn(y,1) for y in layer_sizes[1:]]
    #sigmoid activation function
    def sigmoid(self,z):
        """
        Parameters
        ----------
        z : input of activation function
        Returns
        -------
        act: activation function value
        """
        act = 1.0/(1.0 + np.exp(-z))
        #act=np.exp(z)/(1+np.exp(z))
        return act
    # derivative function of sigmoid activation function
    def sigmoid_prime(self,z):
        """
        Parameters
        ----------
        z : input of activation derivative function
        Returns
        4.3 实验内容 –35/85–
        -------
        act: derivative activation function value
        """
        act = self.sigmoid(z)*(1.0-self.sigmoid(z))
        return act
    # feed forward to get prediction
    def feed_forward(self,x):
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
            output = self.sigmoid(np.dot(w,output)+b)
        return output
    # feed backward to update NN paremeters
    def feed_backward(self,x,y):
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
        #activations of input layer
        activation = np.transpose(x)
        activations = [activation]
        # input after input layer
        layer_input = []
        #forward to get each layer’s input and output
        for b, w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            layer_input.append(z) #input of each layer
            activation = self.sigmoid(z)
            activations.append(activation)#output of each layer
        #loss funtion
        ground_truth = np.transpose(y)
        diff = activations[-1] - ground_truth
        #get input of last layer
        last_layer_input = layer_input[-1]
        delta = np.multiply(diff,self.sigmoid_prime(last_layer_input))
        #bias update of last layer
        delta_b[-1] = np.sum(delta,axis=1,keepdims=True)
        #weight update of last layer
        delta_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        #update weights and bias from 2nd layer to last layer
        for i in range(2,self.num_layers):
            input_values = layer_input[-i]
            delta = np.multiply(np.dot(np.transpose(self.weights[-i+1]),delta),\
            self.sigmoid_prime(input_values))
            delta_b[-i] = np.sum(delta,axis=1,keepdims=True)
            delta_w[-i] = np.dot(delta,np.transpose(activations[-i-1]))
        return delta_b,delta_w
    #training using BP
    def fit(self, x,y,learnrate,mini_batch_size, epochs=1000):
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
        n = len(x)#training size
        for i in range(epochs):
            randomlist = np.random.randint(0,n-mini_batch_size,int(n/mini_batch_size))
            batch_x = [x[k:k+mini_batch_size] for k in randomlist]
            batch_y = [y[k:k+mini_batch_size] for k in randomlist]
            for j in range(len(batch_x)):
                delta_b,delta_w = self.feed_backward(batch_x[j], batch_y[j])
                self.weights = [w - (learnrate/mini_batch_size)*dw for w, dw in
                zip(self.weights,delta_w)]
                self.biases = [b - (learnrate/mini_batch_size)*db for b, db in
                zip(self.biases,delta_b)]
            if (i+1)%100 == 0:
                labels = self.predict(x)
                acc = 0.0
                for k in range(len(labels)):
                    if y[k,labels[k]]==1.0:
                        acc += 1.0
                acc=acc/len(labels)
                print("iterations %d accuracy %.3f"%(i+1,acc))
    #predict function
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
        labels = [np.argmax(results[:,y]) for y in range(results.shape[1])]
        return labels