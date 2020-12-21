import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()

# Store features matrix in x
X = iris.data

# Store target vector in y
y = iris.target

"""
# Features
print(iris.feature_names)

# size of featuer matrix
print(iris.data.shape)

# Size of target vector
print(iris.target.shape)
"""

# import the classifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X, y)

# Predicting for some random values
y_pred = knn.predict(np.array([3, 5, 4, 2]).reshape(1, -1))

print(y_pred)


# Splitting X and Y values in training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(accuracy_score(y_test, y_pred))

# Choosing the best parameter for accuracy

# try K = 1 through K = 30 and record testing accuracy
k_range = list(range(1, 31))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(accuracy_score(y_pred, y_test))

# Visualizing the results
#import matplotlib


# plot the relationship between k and accuaracy score
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

# take k value as 10
