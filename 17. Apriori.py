# Apriori

'''
Association Rule:::::
    *****When to use*****
    - It is used in any dataset where features take only two values, i.e. only 0 & 1 values
        eg.> Market Basket Analysis 
           > People visit X website are likely to visit Y website
           > People having salary in range of (X & Y) salary, are likely to own House
           > People who has watched X Movie, are likely to watch Y Movie
'''



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)

 
'''
Need to create the loop, as we need to predict B, considering A has already happened
'''

transactions = [] # Variable
for i in range(0, 7501): # 7501 is the size of Dataset
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

'''
above appends the **transactions** for values such that range is from rows(0 to 7501)
and columns (0 to 20)
'str' is used, as we need the data in STRING format
'''

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Transaction == is the variable created in loop
# min_support == it is the minimum support of relations. (for a product 3 times a day, 
#                i.e. 21 times a week, but the total number transactions are 7500, so 
#                the min_support is approx 0.003)

# min_confidence == it is minimum confidence of relations.

# min_lift == it is minimum lift of relations.



# Visualising the results
results = list(rules)
print(results)


''' Apriori::
    If A has happened, than there is high likelihood that B would also happen.
    It is a recommendation feature, that is used extensively, to predict the
    the recommendation on basis of the PRIOR knowledge.
    '''
    
'''
Step 1: Set a minimum Support & Confidence
Step 2: Take all subsets in transactions having having higher support than minimum support
Step 3: Take all the rules of these subsets having higher confidence than minimum confidence
Step 4: Sort all rules by decreasing lift
'''