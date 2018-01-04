#Algorithms such as perceptrons are considered to have linear decision boundary.
#The Iris data set I've fetched here have distinct features for separate categories of flowers.
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                  header=None)

df.tail()
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')				  
				
# Just like book, and other videos have mentioned, this Perceptron
# is not complicated idea, we really are just counting errors.
class Perceptron (object):
	
	# eta is a learning rate (why did they name it eta?).
	# n_iter iteration for some sort of training.
	# seeding for random.
	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter=n_iter
		self.random_state=random_state
	
	# X is For z = (some vectors) + ....
	# y is for target values?
	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state) # Random generator, that's really it.
		#According to the book, the perceptron algorithm does not directly co-relate
		#with the normal distribution; however, the line directly below sets 
		#all the weight values to be non-zero.
		self.w_ =  [0.01, 0.2, -0.1]   #rgen.normal(loc=0.0, scale =0.01, size=1+X.shape[1])
		
		print("Weights are: ", self.w_)
		
		self.errors_ = []
		
		#print("At fit")
		#print("X is: ", X)
		#print("y is: ", y)
		#print("w is: ", self.w_[:])
		#input("Press enter to continue...")
		
		#iterate according to the amount of n_iter.
		for epoch in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				#print("In fit for loop")
				#print("xi is: ", xi)
				#print("target is: ", target)
				#The line directly below is delta W.
				update=self.eta * (target - self.predict(xi))
				self.w_[1:] += update*xi
				
				if update != 0:
					print("Miss fire.")
					print("Update is: ", update)
					print("self.w_[1:] now: ", self.w_[1:])
				
				#if update == 0:
				#	print("Not a miss fire.")
				#	print("update is: ", update)
				#	print("Weights are: ", self.w_[1:])	
				
				#The line be directly is related to something called biased-unit.
				self.w_[0] += update 
				#The line below is for the wrong predictions, where the boolean
				#values are represented in 0, or 1.
				errors += int(update != 0.0)
			self.errors_.append(errors) 
		#print("errors_ now: ", self.errors_)
		#print("Returning preceptron: ", self)
		return self
	
	# The below method is what sums up everything regarding linear classification with the perceptron algorithm.
	# After all, dot product is the representation of 
	# This is Z.
	# Machine Learning, 2nd Edition refers this portion as the activation (the firing of the neurons and such). 
	def net_input(self, X):
		#This returns z (that is mentioned in the book), that's about it.
		#print("At net input")
		#print("X is:", X)
		#print("biased value is is:", self.w_[0])
		#print("np.dot is: ", np.dot(X, self.w_[1:]))
		#print("plus after adding biased unit, it's: ", np.dot(X, self.w_[1:]) + self.w_[0])
		#The self.w_[0] represents the w_0.
		return np.dot(X, self.w_[1:]) + self.w_[0]
	
	def predict(self, X):
		#If the first argument is true, return 1, if not,
		#return -1.
		#print("self.net_input is: ", self.net_input(X))
		#print("np.where(self.net_input(X) >= 0.0, 1, -1) is: ", np.where(self.net_input(X) >= 0.0, 1, -1))
		#print("np.where(self.net_input(X) >= 0.0, 1, -1) type is: ", type(np.where(self.net_input(X) >= 0.0, 1, -1)))
		return np.where(self.net_input(X) >= 0.0, 1, -1)
						
#The below lines of code is a demonstration that linear classification is possible for Iris data set.			
#plt.scatter(X[:50, 0], X[:50, 1],
#			color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1],
#            color='blue', marker='x', label='versicolor')
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')
#plt.show()
# The above lines of code is a demonstration that linear classification is possible for Iris data set.

#print("Innitializing X & y")
#print("X is: ", df.iloc[0:100, [0, 2]].values)
X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
#Squash the two types of data types into -1 or 1.
#Where Iris-setosa is -1, and whatever other is 1.
#print("y is: ", np.where(y== 'Iris-setosa', -1, 1))
y = np.where(y == 'Iris-setosa', -1, 1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
#plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#plt.xlabel('Epochs')
#plt.ylabel('Number of updates')
#plt.show()

# The below method is really just for the graphing purpose.
def plot_decision_regions(X, y, classifier, resolution=0.02):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	
	print("x1_min is: ", x1_min) 
	print("x1_max is: ", x1_max)
	print("x2_min is: ", x2_min)
	print("x2_max is: ", x2_max)
	
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	
	print("xx1 is: ", xx1)
	print("xx2 is: ", xx2)
	
	#ravel() eliminates the multi-dimensionalities, where the returned array is always one-dimensional array.
	#In this case, classifier is the perceptron algorithm which is already has the adjusted weights.
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	print("xx1 ravel is: ", xx1.ravel())
	print("xx2 ravel is: ", xx2.ravel())
	#print("np.array([xx1.ravel(), xx2.ravel()]) is: ", np.array([xx1.ravel(), xx2.ravel()]))
	#In all obviousnes, the .T invocation is for the transpose.
	#print("np.array([xx1.ravel(), xx2.ravel()]).T is: ", np.array([xx1.ravel(), xx2.ravel()]).T)
	print("Z is: ", Z)
	
	Z = Z.reshape(xx1.shape)
	print("Z.reshape(xx1.shape) is: ", Z)
	
	#The contourf is what's making the linear distinction here.
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	
	# plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
			
plot_decision_regions(X, y, classifier=ppn)
plt.show()