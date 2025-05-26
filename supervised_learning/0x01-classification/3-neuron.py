#!/usr/bin/env python3
"""
Write a class Neuron that defines a single neuron performing binary classification (Based on 2-neuron.py):

Add the public method def cost(self, Y, A):
   Calculates the cost of the model using logistic regression
   Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
   A is a numpy.ndarray with shape (1, m) containing the activated output of the neuron for each example
   To avoid division by zero errors, please use 1.0000001 - A instead of 1 - A
   Returns the cost
alexa@ubuntu-xenial:$ cat 3-main.py
#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
A = neuron.forward_prop(X)
cost = neuron.cost(Y, A)
print(cost)
alexa@ubuntu-xenial:$ ./3-main.py
4.365104944262272
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
        Calculates the cost of the model using logistic regression
        Arguments:
         - Y (numpy.ndarray): is a numpy.ndarray with shape (1, m) that
           contains the correct labels for the input data
         - A is a numpy.ndarray with shape (1, m) containing the activated
           output of the neuron for each example
        Returns:
         The cost
        """
        m = Y.shape[1]
        cost = - (1/m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost
