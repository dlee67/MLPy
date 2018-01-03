

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
		self.w_ = rgen.normal(loc=0.0, scale =0.01, size=1+X.shape[1])
		self.errors_ = []
		
		print("y is: ", y)
		
		for _ in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				#Most likely, the below lines are about delta W.
				update=self.eta * (target - self.predict(xi))
				self.w_[1:] += update*xi
				#The line be directly is related to something called biased-unit.
				self.w_[0] += update 
				#The line below is for the wrong predictions, where the weights are pushed towards the
				#direction of the positive or negative class.
				errors += int(update != 0.0)
				print("After the evaluation:...",)
				print("xi is: ", xi)
				print("target is: ", target)
				print("update is: ", update)
				print("_ is:", _)
				print("self.w_ is", self.w_[:])
				print("errors is: ", errors)
				input("Press enter to continue...")
			#self.errors_.append(errors)
		return self
	
	def net_input(self, X):
		#This returns z (that is mentioned in the book), that's about it.
		return np.dot(X, self.w_[1:]) + self.w_[0]
	
	def predict(self, X):
		#If the first argument is true, return 1, if not,
		#return -1.
		return np.where(self.net_input(X) >= 0.0, 1, -1)
			
			
# The below lines of code is a demonstration that linear classification is possible for Iris data set.			
			
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
#
X = df.iloc[0:100, [0, 2]].values
#
#plt.scatter(X[:50, 0], X[:50, 1],
#			color='red', marker='o', label='setosa')
#plt.scatter(X[50:100, 0], X[50:100, 1],
#            color='blue', marker='x', label='versicolor')
#plt.xlabel('sepal length [cm]')
#plt.ylabel('petal length [cm]')
#plt.legend(loc='upper left')
#plt.show()

# The above lines of code is a demonstration that linear classification is possible for Iris data set.

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
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	# plot class samples
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')
			
plot_decision_regions(X, y, classifier=ppn)
plt.show()