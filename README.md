# HW1_70074425_TUMMALA
Implementing Naïve Bayes method, linear SVM method and Linear Regression using scikit-learn

# Neural Networks & Deep Learning

Neural networks are a type of machine learning algorithm that is inspired by the human brain. Deep learning is a subset of machine learning that uses neural networks to solve complex problems.

Neural networks are made up of interconnected nodes, which are similar to the neurons in the human brain. These nodes are arranged in layers, and each layer performs a different function. The first layer takes in the input data, and the last layer outputs the predictions. The intermediate layers learn to extract features from the input data, which are then used by the last layer to make predictions.

Deep learning is a type of machine learning that uses neural networks with many layers. These deep neural networks are able to learn complex relationships between the input data and the output predictions. This makes them well-suited for solving problems that are difficult for traditional machine learning algorithms to solve.

Here are some of the benefits of using neural networks and deep learning:
--> They can learn complex relationships between the input data and the output predictions.
--> They can be used to solve problems that are difficult for traditional machine learning algorithms to solve.
--> They are becoming increasingly powerful and sophisticated.

However, there are also some challenges associated with using neural networks and deep learning:
--> They can be computationally expensive to train.
--> They can be difficult to interpret.
--> They can be susceptible to overfitting.

Overall, neural networks and deep learning are powerful tools that can be used to solve a wide variety of problems. However, it is important to be aware of the challenges associated with using these techniques before using them.

# Naïve Bayes Method:

Naïve Bayes is a classification algorithm based on Bayes' theorem with the assumption of independence between features. It is a simple yet effective probabilistic classifier that is commonly used for text classification and spam filtering. 

The Naïve Bayes algorithm calculates the probability of a given data point belonging to a specific class based on the probabilities of its features. It assumes that the features are conditionally independent given the class label, which is why it is called "naïve."

# Naive Bayes Classification on the Glass Dataset:
This repository contains code for implementing a Naive Bayes classifier on the glass dataset. The dataset contains attributes of glass samples, and the goal is to predict the type of glass based on these attributes.
--> The code is written in Python and uses the scikit-learn library. The following steps are performed:
--> The glass dataset is loaded.
--> The data is split into training and testing sets.
--> A Gaussian Naive Bayes classifier is created.
--> The classifier is fit to the training data.
--> The classifier is used to predict the class labels for the test data.
--> The accuracy of the classifier is calculated.
--> A classification report is printed.

In order to run the code, we need to have Python and the scikit-learn library installed. we can then clone the repository and run the following command:

# python code: NaiveBayes.py

This will run the code and print the accuracy of the classifier and the classification report.

Dependencies
Python 3.6+
scikit-learn


# Linear SVM Method:

Linear SVM (Support Vector Machine) is a classification algorithm that separates classes by finding the optimal hyperplane that maximizes the margin between the classes. It is a powerful and widely used algorithm for binary classification problems.

# Linear SVM Classification on the Glass Dataset 

This repository contains code for implementing a linear SVM classifier on the glass dataset. The dataset contains attributes of glass samples, and the goal is to predict the type of glass based on these attributes.

The code is written in Python and uses the scikit-learn library. The following steps are performed:
--> The glass dataset is loaded.
--> The data is split into training and testing sets.
--> A linear SVM classifier is created.
--> The classifier is fit to the training data.
--> The classifier is used to predict the class labels for the test data.
--> The accuracy of the classifier is calculated.
--> A classification report is printed.

In order to run the code, we need to have Python and the scikit-learn library installed. we can then clone the repository and run the following command:

# python code:  Linear_SVM_Method.py

This will run the code and print the accuracy of the classifier and the classification report.

Dependencies
Python 3.6+
Scikit-learn



# Which algorithm got better accuracy? Can you justify why?

After implementing both the methods, we know that the output of the code will be the accuracy of the two classifiers. In this case, the accuracy of the linear SVM classifier is 0.56, while the accuracy of the Naive Bayes classifier is 0.52. This means that the linear SVM classifier is slightly more accurate than the Naive Bayes classifier.

There are a few reasons why the linear SVM classifier might have been more accurate than the Naive Bayes classifier. First, the linear SVM classifier is a more powerful model than the Naive Bayes classifier. Second, the linear SVM classifier is able to learn non-linear relationships between the features and the labels, while the Naive Bayes classifier can only learn linear relationships.

However, it is important to note that the accuracy of a classifier can vary depending on the dataset. In other words, the linear SVM classifier might not be more accurate than the Naive Bayes classifier on a different dataset.

# Linear Regression:

Linear regression is a statistical method that uses a linear model to predict the value of a dependent variable from one or more independent variables. The linear model is a line that best fits the data points. The dependent variable is the variable that we are trying to predict, and the independent variables are the variables that we are using to predict the dependent variable.

The linear regression model is trained by finding the line that best fits the data points. The line that best fits the data points is the line that minimizes the sum of the squared errors. The squared errors are the distances between the data points and the line.

Once the linear regression model is trained, it can be used to predict the value of the dependent variable for new data points. The predicted value is the value of the line that is closest to the new data point.

# Linear Regression on Salary Data

This repository contains code for implementing a linear regression model on the Salary_Data.csv dataset. The dataset contains the years of experience and salaries of a group of employees. The goal is to predict the salary of an employee given their years of experience.

The code is written in Python and uses the scikit-learn library. The following steps are performed:
--> The Salary_Data.csv file is imported.
--> The data is split into train and test partitions.
--> A linear regression model is created.
--> The model is trained on the training data.
--> The salaries for the test data are predicted.
--> The mean squared error is calculated.
--> The train and test data are visualized using a scatter plot.

In order to run the code, we will need to have Python and the scikit-learn library installed. we can then clone the repository and run the following command:

# python code: program-3 LR.py

This will run the code and print the mean squared error, as well as a scatter plot of the predicted salaries and the actual salaries.
Dependencies
Python 3.6+
scikit-learn

In the output we observe that It is slightly less accurate on the test data than it is on the train data. However, It is still relatively accurate, and the scatter plot shows that the predicted salaries are generally close to the actual salaries.


# Youtube video link:
https://youtu.be/JSSkF_-M2l8


















