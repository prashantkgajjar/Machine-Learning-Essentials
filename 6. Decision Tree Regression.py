# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
'''
for decison tree, by default a decision tree would calculate on basis of Mean Square Error
'''

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])


#Visualizing the Decision Tree (Defaul Plots)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression - DEFAULT)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


# Visualising the Decision Tree Regression results (higher resolution)
'''
As can be seen from the plot, the decision tree shows the plot which is similar to Polynomial
Regression.
The plot is taking the route of Gradient Descent.
However, in general, it should be average between two value, as the Decision Tree is a 
NON CONTINOUS MODEL
A decision tree should have plot, which could clearly showcase the trees / branches in the 
dataset.
'''



X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

'''
As can be seen from the plot, the decision tree now showcase the different intervals, and it
is now showing the average of each of the element.
'''