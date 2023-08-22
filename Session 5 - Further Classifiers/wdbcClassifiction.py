
"""
@author: jacob
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#load the dataset
wdbc = datasets.load_breast_cancer()

X = wdbc.data
y = wdbc.target


#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#For the following models, some experimentation has been performed
#in order to find the appropriate parameter for the models.
#This isn't given here, but you might try varying the values.
#You might well want to plot the curves that your find.

#set up kNN
knn_model = KNeighborsClassifier(n_neighbors=12)

#set up randon forest
rf_model = RandomForestClassifier(n_estimators = 15)

#set up naive Bayes
gnb_model = GaussianNB()

#fit and test kNN
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))

#fit and test kNN
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('RF Accuracy: %.3f' % accuracy_score(y_test, y_pred_rf))

#fit and test kNN
gnb_model.fit(X_train, y_train)
y_pred_gnb = gnb_model.predict(X_test)
print('GNB Accuracy: %.3f' % accuracy_score(y_test, y_pred_gnb))

