import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()

# Store feature matrix in X
X = iris.data

# Store target vector in y
y = iris.target

knn = KNeighborsClassifier(n_neighbors=1)

# Training the model with X and y
knn_clf = knn.fit(X, y)

# Saving the model as a picle in a file
joblib.dump(knn_clf, "Knn_Classifier.pkl")
