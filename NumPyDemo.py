from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#rand = np.random.randint(100)
#
#for x in range(100):
#	print(rand)
#	rand = np.random.randint(100)
	
arr1 = np.array([[1, 2], [1, 2], [1, 2]])
arr2 = np.array([[1, 2, 4], [1, 2, 4]])

mat1 = np.matrix(arr1)
mat2 = np.matrix(arr2)

print(mat1.sum())
print(np.reshape(arr1, (2, 3)))
print("----------------------------------")

rand = np.random.rand(2, 2)
print(rand)
print("----------------------------------")
modified = np.random.rand(250 * 10).reshape((250, 10)) * 10
print(modified)
print("------------About random------------------")
rand_gen = np.random.RandomState()
print(rand_gen.rand())
print("Gaussian distribution: ", rand_gen.normal(loc=0.0, scale=0.01, size=1)) # normal() generates a Gaussian distribution (normal distribution).


