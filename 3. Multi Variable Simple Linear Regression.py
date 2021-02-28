# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:47:14 2020

@author: Prashant
"""
# Import necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Dataset

df = pd.read_csv('50_Startups.csv')

'''
As can be seen from dataset, we need to assign numerical value in "Country" column for
better data modelling
'''

df_dummies = pd.get_dummies(df['State'])

# Now, there are '3 columns'. So, we need to get rid of last column to get rid of 
#"Dummy Variable Trap"

df_dummies = df_dummies.drop('New York', axis = 'columns')

#Now as our dummies are ready, now need to merge the data.

df = pd.concat([df, df_dummies], axis = 'columns')

# Need to assume X

X = df.drop(['State', 'Profit'], axis = 'columns')

# We need to predict profit from the different categories. So, "Profit" would be Y Dataset

y = df['Profit']

# Now, as our X & Y are ready, we need to fit the model.

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Also, we need to slpit the data for training & testing data.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8)

# Fitting the model
lr.fit(X_train, y_train)

# Model Accuracy
lr.score(X_test, y_test)

# Model Prediction of Test Dataset
y_pred = lr.predict(X_test)

# Optimizing the model using BACKWORKD ELIMIATION METHOD
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1), dtype = int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
ols = sm.OLS(endog = y, exog = X_opt).fit() 
ols.summary() 