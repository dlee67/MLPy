#When it comes to algorithms, whoever is teaching it will not tell us the significances within the algorithm,
#that makes it work.
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
	
	def __init__(self, eta=0.01, n_iter=50, random_state=1):
		self.eta = eta
		self.n_iter=n_iter
		self.random_state=random_state
		self.final_weight = None
	
	def fit(self, X, y):
		rgen = np.random.RandomState(self.random_state) # Random generator, that's really it.
		self.w_ = rgen.normal(loc=0.0, scale =0.01, size=1+X.shape[1])
		self.errors_ = []
		#iterate according to the amount of n_iter.
		for epoch in range(self.n_iter):
			errors = 0
			for xi, target in zip(X, y):
				update=self.eta * (target - self.predict(xi)) #Still bit confused about why there is nothing here that prepresents the X^i	
				if update != 0:
					print("Miss fire on target: ", target, " ,with data: ", xi)
					print("Weight now: ", self.w_[1:])
					print("Biased unit at: ", self.w_[0])
					print("Update's at: ", update)
					print("net_input returns: ", self.net_input(xi))
					print("Weight later: ", (self.w_[1:] + update*xi))
					print("\n")
					
				#if update == 0:
				#	print("No miss fire on target: ", target, " ,with data: ", xi)
				#	print("Weight now: ", self.w_[1:])
				#	print("Biased unit at: ", self.w_[0])
				#	print("Update's at: ", update)
				#	print("net_input returns: ", self.net_input(xi))
				#	print("Weight later: ", (self.w_[1:] + update*xi))
				#	print("\n")
				
				self.w_[1:] += update*xi			
				self.w_[0] += update 
				errors += int(update != 0.0)
			self.errors_.append(errors) 
		return self
	
	# https://math.stackexchange.com/questions/1461038/how-exactly-does-the-sign-of-the-dot-product-determine-the-angle-between-two-vec
	# Everything is explained in the above link.
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
	
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

X = df.iloc[0:100, [0, 2]].values
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
	
def graph(formula, slope, intercept):
	x = np.arange(10)
	y = formula(slope, x, intercept)
	plt.plot(x, y)
	plt.show()
	
def lin_eq(m, x, b):
	return m*x + b
	
plot_decision_regions(X, y, classifier=ppn)
plt.show()