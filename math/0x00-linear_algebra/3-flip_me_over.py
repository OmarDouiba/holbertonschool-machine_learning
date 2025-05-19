#!/usr/bin/env python3
"""
task:

Write a function def matrix_transpose(matrix): that returns the transpose of a 2D matrix, matrix:

You must return a new matrix
You can assume that matrix is never empty
You can assume all elements in the same dimension are of the same type/shape
"""

def matrix_transpose(matrix):
    """function that returns the transpose of a matrix"""
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]