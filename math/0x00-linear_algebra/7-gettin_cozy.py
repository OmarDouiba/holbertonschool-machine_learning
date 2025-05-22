#!/usr/bin/env python3
"""
Write a function def cat_matrices2D(mat1, mat2, axis=0): that concatenates two matrices along a specific axis:

You can assume that mat1 and mat2 are 2D matrices containing ints/floats
You can assume all elements in the same dimension are of the same *type/shape
You must return a new matrix
If the two matrices cannot be concatenated, return None
alexa@ubuntu-xenial:0x00-linear_algebra$ cat 7-main.py 
#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print(mat4)
print(mat5)
mat1[0] = [9, 10]
mat1[1].append(5)
print(mat1)
print(mat4)
print(mat5)
alexa@ubuntu-xenial:0x00-linear_algebra$ ./7-main.py 
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
[[9, 10], [3, 4, 5]]
[[1, 2], [3, 4], [5, 6]]
[[1, 2, 7], [3, 4, 8]]
alexa@ubuntu-xenial:0x00-linear_algebra$ 
"""

def cat_matrices2D(mat1, mat2, axis=0):
    """
    Function that concatenates two matrices along a specific axis.

    Parameters:
    - mat1 (list of lists of ints/floats): 2D matrix to concatenate.
    - mat2 (list of lists of ints/floats): 2D matrix to concatenate.
    - axix (int): axis to concatenate.

    Return:
     A new matrix that concatenates the matices,
     if the two matices cannot be concatenated, return None.
    """
    if not mat1 or not mat2:
        return None
    
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        # row[:] creates a shallow copy of each row
        # so changes to the original matrix won’t affect the new one.
        return [row[:] for row in mat1] + [row[:] for row in mat2]
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        return [(mat1[i] + mat2[i]) for i in range(len(mat1))]

    return None
