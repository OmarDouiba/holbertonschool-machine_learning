#!/usr/bin/env python3


"""
Write a function def cat_arrays(arr1, arr2): that concatenates two arrays:

You can assume that arr1 and arr2 are lists of ints/floats
You must return a new list
alexa@ubuntu-xenial:0x00-linear_algebra$ cat 6-main.py 
#!/usr/bin/env python3

cat_arrays = __import__('6-howdy_partner').cat_arrays

arr1 = [1, 2, 3, 4, 5]
arr2 = [6, 7, 8]
print(cat_arrays(arr1, arr2))
print(arr1)
print(arr2)
alexa@ubuntu-xenial:0x00-linear_algebra$ ./6-main.py 
[1, 2, 3, 4, 5, 6, 7, 8]
[1, 2, 3, 4, 5]
[6, 7, 8]
alexa@ubuntu-xenial:0x00-linear_algebra$ 
"""

def cat_arrays(arr1, arr2):
    """function that concatenates two arrays"""
    return arr1 + arr2