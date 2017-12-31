#This entire program is produced while referring to Introduction to Machine Learning with Python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'])
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print("Iris data: {}".format(iris_dataset['data']))
print("---------------------------------------------------------------------------")

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_train shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_test.shape))