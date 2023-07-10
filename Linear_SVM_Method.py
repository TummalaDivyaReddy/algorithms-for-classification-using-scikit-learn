import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

# Loading the glass dataset
data = pd.read_csv('glass.csv')

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Type', axis=1), data['Type'], test_size=0.25)

# Creating a linear SVM classifier
clf = LinearSVC()

# Fitting the classifier to the training data
clf.fit(X_train, y_train)

# Predicting the class labels for the test data
y_pred = clf.predict(X_test)

# Calculating the accuracy of the model
accuracy = clf.score(X_test, y_test)

print('Accuracy:', accuracy)

# Printing the classification report
print(classification_report(y_test, y_pred))