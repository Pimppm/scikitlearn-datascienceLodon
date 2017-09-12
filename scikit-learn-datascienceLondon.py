# -*- coding: utf-8 -*-
"""
Scikits-learn machine learning in Python
data sets provided by Data Science London through kaggle.com
@author: pim
"""

#import required modules and data sets
import numpy as np # array
import pandas as pd #Data structures and analysis

from sklearn.utils import shuffle

import os

#set work directory
os.chdir('C:/Users/pim/MSBI/Competition/Data Science London')


#read in data sets
train = pd.read_csv('train.csv', header=None)
trainLabels = pd.read_csv('trainLabels.csv', header=None)
test = pd.read_csv('test.csv', header=None)

#look overall of the data sets
train.describe()
trainLabels.describe()
test.describe()

# generate asarray
train = np.asarray(train)
trainLabels = np.asarray(trainLabels).ravel() # a 1-D array .ravel() 
test = np.asarray(test)

print (train.shape, trainLabels.shape, test.shape)

"""
Building Models
"""
from sklearn.model_selection import train_test_split
# split training data into two sets - 40% test, 60% train 
X_train, X_test, y_train, y_test = train_test_split(train, trainLabels, test_size=0.4, random_state=4)

#let check the shape of the new objects
print ('Training set')
print (X_train.shape, y_train.shape)
print ('Testing set')
print (X_test.shape, y_test.shape)

'''
LogisticRegression
'''
from sklearn.linear_model import LogisticRegression

#training the model on the training set
logreg = LogisticRegression()

# fit the model with the data
logregmodel = logreg.fit(X_train, y_train)

#make predictions on the testing set
logregpred = logreg.predict(X_test)

# cross-validation
from sklearn import metrics

print ('Accuracy')
print(metrics.accuracy_score(y_test, logregpred))

'''
KNeighbor
'''
from sklearn.neighbors import KNeighborsClassifier

'K = 5'
knn5 = KNeighborsClassifier(n_neighbors=5)

#fit model to the data
knn5model = knn5.fit(X_train, y_train)

knn5pred = knn5.predict(X_test)

# compute classification accuracy
print ('Accuracy')
print (metrics.accuracy_score(y_test, knn5pred))

'K = 1'
knn = KNeighborsClassifier(n_neighbors=1)

#fit model to the data
knn1model = knn.fit(X_train, y_train)

knn1pred = knn.predict(X_test)

# compute classification accuracy
print ('Accuracy')
print (metrics.accuracy_score(y_test, knn1pred))

# figure out value for K
# try K-1 through K-10 and record testing accuracy

k_range = range(1,10)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knnpred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, knnpred))
    
import matplotlib.pyplot as plt # plotting library

plt.plot(k_range,scores)
plt.xlabel('Value of K for knn')
plt.ylabel('Testing Acuracy')

' Conclusion: the best K is 4 for this data set. '

"""
Makeing predictions on test sample set
"""
knn4 = KNeighborsClassifier(n_neighbors=4)

# train the model
knn4model = knn4.fit(train, trainLabels)

# make a prediction for test set
knn4pred = knn4.predict(test)

print (knn4pred)
