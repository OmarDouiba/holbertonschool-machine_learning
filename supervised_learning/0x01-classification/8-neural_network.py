#!/usr/bin/env python3
"""
Defines a NeuralNetwork class for binary classification
"""
import numpy as np

class NeuralNetwork:
    """
    Neural network with one hidden layer for binary classification
    """

    def __init__(self, nx, nodes):
        """
        Class constructor

        Parameters:
        nx (int): Number of input features
        nodes (int): Number of nodes in the hidden layer

        Raises:
        TypeError: If nx or nodes is not an integer
        ValueError: If nx or nodes is less than 1
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # Hidden layer weights and bias
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = 0
        self.A1 = 0

        # Output neuron weights and bias
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
