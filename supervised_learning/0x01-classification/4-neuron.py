#!/usr/bin/env python3
"""
Write a class Neuron that defines a single neuron performing binary classification (Based on 3-neuron.py):

Add the public method def evaluate(self, X, Y):
   Evaluates the neuron’s predictions
   X is a numpy.ndarray with shape (nx, m) that contains the input data
       nx is the number of input features to the neuron
       m is the number of examples
   Y is a numpy.ndarray with shape (1, m) that contains the correct labels for the input data
   Returns the neuron’s prediction and the cost of the network, respectively
       The prediction should be a numpy.ndarray with shape (1, m) containing the predicted labels for each example
       The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise
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
    
        