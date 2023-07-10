import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
datasets = pd.read_csv('Salary_Data.csv')
X = datasets.iloc[:, :-1].values
Y = datasets.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

# Predicting the Test set result
Y_Pred = regressor.predict(X_Test)

# Visualizing the Training set results
plt.scatter(X_Train, Y_Train, color='red', label='Training Data')
plt.plot(X_Train, regressor.predict(X_Train), color='blue', label='Linear Regression')
plt.title('Training set')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Visualizing the Test set results
plt.scatter(X_Test, Y_Test, color='red', label='Test Data')
plt.plot(X_Train, regressor.predict(X_Train), color='blue', label='Linear Regression')
plt.title('Test set')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
