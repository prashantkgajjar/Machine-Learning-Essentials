# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

'''
To avoid the double quotes in the dataset, quoting = 3 is used to do that...

'''
'''
        re.sub = removes the characters
        [^a-zA-Z] = do not review any characters
        dataset['Review'][i] = considers the characters from column i only
        ' ' = considers the characters with space< > in it
'''
'''
        Above firstly creates FOR LOOP.
        secondly it creates IF Condition
        set() = it is an argument, & would make process much quicker
'''

# Cleaning the texts
import re #
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # Imports stopwords package
from nltk.stem.porter import PorterStemmer 
'''used to  remove the commoner morphological and inflexional endings'''
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    review = review.lower() # converts the capital letters with lower case letters.
    review = review.split() # splits the reviews in different parts inorder to make a list
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]  
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
'''
Creating the number of columns for the corresponding words
'''

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


'''
55 correct predictions of negative reviews
91 correct prediction of positive reviews

12 incorrect predictions of negative reviews
42 incorrect predictions of positive reviews
'''

classifier.score(X_test, y_test)
# the model accuracy is 73%!!!