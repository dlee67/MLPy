import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func(x, a, b):
	print("Passed in: ", x)
	print("Returning: ", a * x + b)
	return a * x + b
	
x = np.linspace(0, 10, 100)
y = func(x, 1, 2)

yn = y + 0.9 * np.random.normal(size=len(x))
popt, pcov = curve_fit(func, x, yn)

#print(np.linspace(0, 100, 100))
#some_array = np.array([1, 2, 3])
#print(some_array * 2)
