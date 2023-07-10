import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Load the glass dataset
data = pd.read_csv('glass.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('Type', axis=1), data['Type'], test_size=0.25)

# Create a Gaussian Naïve Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train, y_train)

# Predict the class labels for the test data
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy using Naïve Bayes:", accuracy)

# Print the classification report
report = classification_report(y_test, y_pred)
print("Classification Report for Naïve Bayes:")
print(report)
