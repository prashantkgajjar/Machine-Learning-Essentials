## Simple Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Importing the DataSets

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:, 0:1].values
y = df.iloc[:, -1:].values

# Training the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

X_train.shape
X_test.shape

'''
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
'''

##### Using Regression Technique

# Impoting the Regression Techniques
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Fitting the model
lr.fit(X_train, y_train)

# predicting the model
y_pred = lr.predict(X_test)

# Accuracy of the model
lr.score(X_test, y_test)

##### Visualization

#Training Visualization
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lr.predict(X_train), color = 'blue')
plt.title('Salary v/s Expereice')
plt.xlabel('Experience in Years')
plt.ylabel('Salary')
plt.show()

#Testing Visualization
plt.scatter(X_test, y_test, color = 'yellow')
plt.plot(X_test, lr.predict(X_test), color = 'blue')
plt.title('Salary v/s Expereice')
plt.xlabel('Experience in Years')
plt.ylabel('Salary')
plt.show()
