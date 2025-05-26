#!/usr/bin/env python3
"""
Write a class Neuron that defines a single neuron performing binary classification (Based on 5-neuron.py):

Add the public method def train(self, X, Y, iterations=5000, alpha=0.05):
   Trains the neuron
   X is a numpy.ndarray with shape (nx, m) that contains the input data
       nx is the number of input features to the neuron
       m is the number of examples
   Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
   iterations is the number of iterations to train over
       if iterations is not an integer, raise a TypeError with the exception iterations must be an integer
       if iterations is not positive, raise a ValueError with the exception iterations must be a positive integer
   alpha is the learning rate
       if alpha is not a float, raise a TypeError with the exception alpha must be a float
       if alpha is not positive, raise a ValueError with the exception alpha must be positive
   All exceptions should be raised in the order listed above
   Updates the private attributes __W, __b, and __A
   You are allowed to use one loop
   Returns the evaluation of the training data after iterations of training have occurred
alexa@ubuntu-xenial:$ cat 6-main.py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('6-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X_train.shape[0])
A, cost = neuron.train(X_train, Y_train, iterations=10)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))
A, cost = neuron.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
alexa@ubuntu-xenial:$ ./6-main.py
Train cost: 1.3805076999077135
Train accuracy: 64.73746545598105%
Dev cost: 1.4096194345468178
Dev accuracy: 64.49172576832152%

Not that great… Let’s get more data!
"""
import numpy as np

class Neuron:
    """
    Class Neuron that defines a simple neuron performing binary classification
    """
    def __init__(self, nx):
        """
        Constructor for the class
        Arguments:
         - nx (int): is the number of input features to the neuron
        Public instance attributes:
         - W: The weights vector for the neuron. Upon instantiation, it should
              be initialized using a random normal distribution.
         - b: The bias for the neuron. Upon instantiation, it should be
              initialized to 0.
         - A: The activated output of the neuron (prediction). Upon
              instantiation, it should be initialized to 0.

        """
        if type(nx) is not int:
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0
        

    @property
    def W(self):
        """
        getter function for W
        Returns weights
        """
        return self.__W
    @property
    def b(self):
        """
        getter gunction for b
        Returns bias
        """
        return self.__b
    @property
    def A(self):
        """
        getter function for A
        Returns activation values
        """
        return self.__A
    
    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Arguments:
        - X (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute __A
        """
        Z = np.dot(self.__W, X) + self.__b
        self.__A = self.sigmoid(Z)
        return self.__A

    def sigmoid(self, Z):
        """
        Applies the sigmoid activation function
        Arguments:
        - z (numpy.ndattay): with shape (nx, m) that contains the input data
         * nx is the number of input features to the neuron.
         * m is the number of examples
        Updates the private attribute __A
        The neuron should use a sigmoid activation function
        Return:
        The private attribute A
        """
        y_hat = 1 / (1 + np.exp(-Z))
        return y_hat
    
    def cost(self, Y, A):
        """
        Calculates the cost using logistic regression loss (cross-entropy)

        Parameters:
        - Y: numpy.ndarray of shape (1, m), true labels
        - A: numpy.ndarray of shape (1, m), predicted activations from sigmoid

        Returns:
        - cost: float, the logistic regression cost
        """
        m = Y.shape[1]
        cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron's predictions

        Parameters:
        - X: numpy.ndarray of shape (nx, m), input data
        - Y: numpy.ndarray of shape (1, m), true labels

        Returns:
        - A tuple: (prediction, cost)
        - prediction: numpy.ndarray of shape (1, m) with predicted labels (0 or 1)
        - cost: float, the cost of the network
        """

        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0) , cost
    
    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron

        Parameters:
        - X: numpy.ndarray of shape (nx, m) with the input data
        - Y: numpy.ndarray of shape (1, m) with correct labels
        - A: numpy.ndarray of shape (1, m) with activated output (predictions)
        - alpha: learning rate (float), default is 0.05

        Updates:
        - self.__W and self.__b using the gradient descent update rule
        """

        dZ = A - Y
        m = X.shape[1]
        dW = np.dot(dZ, X.T) / m
        db = np.sum(dZ) / m

        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
    
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neuron using gradient descent

        Parameters:
        - X: numpy.ndarray of shape (nx, m), input data
        - Y: numpy.ndarray of shape (1, m), correct labels
        - iterations: number of iterations to train (default: 5000)
        - alpha: learning rate (default: 0.05)

        Returns:
        - The evaluation of the training data after the final iteration
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        for i in range(iterations):
            self.__A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
        
        return self.evaluate(X, Y)