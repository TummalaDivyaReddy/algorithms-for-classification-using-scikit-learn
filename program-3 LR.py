import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Importing the Salary_Data.csv file
data = pd.read_csv('Salary_Data.csv')

# Splitting the data into train and test partitions
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Creating a linear regression model
regressor = LinearRegression()

# Training the model
regressor.fit(X_train, y_train)

# Predicting with the model
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Calculating the mean squared error
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

# Visualizing the train and test data using a scatter plot
plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_test, y_test, color='red', label='Test Data')
plt.plot(X_train, y_pred_train, color='black', label='Linear Regression')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression - Train and Test Data')
plt.legend()
plt.show()
