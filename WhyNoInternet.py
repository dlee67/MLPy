# Line 26, weight_vector referenced before the assignment.

from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest import TestCase as ut

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                  header=None)
				  
inputs = df.iloc[0:100, [0, 2]].values
classes = df.iloc[0:100, 4].values
classes = np.where(classes == 'Iris-setosa', -1, 1)
learning_rate = 0.22
rgen = np.random.RandomState(1)
weight_vector = rgen.normal(loc=0.0, scale =0.01, size=1+inputs.shape[1])

def close_to_target(input, current_weight):
	#print("At ctt, returning: ", np.where(np.dot(input, current_weight) >= 0, 1, -1))
	return  np.where(np.dot(input, current_weight) >= 0, 1, -1)

def delta_weight(input, current_weight, target_value):
	return learning_rate*(target_value-(close_to_target(input, current_weight)))
	
def mutate_weight(input, current_weight, target_value):
	weight_vector += delta_weight(input, current_weight, target_value)
	
def mutate_bias(input, current_weight, target_value):
	weight_vector[0] += delta_weight(input, current_weight, target_value)
	
def perceptron(weight):
	weight_now = weight
	epoch = 10
	for iter in range(epoch):
		for current_data, target_data in zip(inputs, classes):
			mutate_weight(current_data, weight_now, target_data)
			mutate_bias(current_data, weight_now, target_data)
	print("Weight now: ", weight_vector)	
		
		
perceptron(weight_vector)