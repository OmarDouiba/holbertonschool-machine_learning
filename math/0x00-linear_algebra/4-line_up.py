#!/usr/bin/env python3
"""
Write a function def add_arrays(arr1, arr2): that adds two arrays element-wise:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list
If arr1 and arr2 are not the same shape, return None
alexa@ubuntu-xenial:0x00-linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
alexa@ubuntu-xenial:0x00-linear_algebra$ ./4-main.py 
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None
alexa@ubuntu-xenial:0x00-linear_algebra$ 
"""

def add_arrays(arr1, arr2):
    """
    Function that add two arrays elemet-wise.

    Parameters:
    - arr1 (list of ints/floats): first list.
    - arr2 (list of ints/floats): second list.

    Return:
     A new list with the first list added second list,
     if the shape of the lists are not the same, return None.
    """
    if len(arr1) == len(arr2):
        return [ (arr1[i] + arr2[i]) for i in range(len(arr1))]
    else:
        return None