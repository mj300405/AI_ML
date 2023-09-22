import numpy as np
from KNN import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=69)

model = KNN(k=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = np.sum(predictions == y_test) / len(y_test)
print('Accuracy for k = 3: ' + str(accuracy))

model = KNN(k=5)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)

print('Accuracy for k = 5: ' + str(accuracy))