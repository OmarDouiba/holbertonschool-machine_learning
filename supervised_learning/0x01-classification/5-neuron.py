#!/usr/bin/env python3
"""
Write a class Neuron that defines a single neuron performing binary classification (Based on 3-neuron.py):
Write a class Neuron that defines a single neuron performing binary classification (Based on 4-neuron.py):

Add the public method def gradient_descent(self, X, Y, A, alpha=0.05):
   Calculates one pass of gradient descent on the neuron
   X is a numpy.ndarray with shape (nx, m) that contains the input data
       nx is the number of input features to the neuron
       m is the number of examples
   Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
   A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
   alpha is the learning rate
   Updates the private attributes __W and __b
alexa@ubuntu-xenial:$ cat 5-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('5-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
neuron.gradient_descent(X, Y, A, 0.5)
print(neuron.W)
print(neuron.b)
alexa@ubuntu-xenial:$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
alexa@ubuntu-xenial:$
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