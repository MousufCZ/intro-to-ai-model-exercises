#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 13:59:52 2021

@author: jacob
"""

#libraries for the task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#load data
iris = datasets.load_iris()

#assign  and y
X = iris.data
y = iris.target
variety = iris.target_names

#split the data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#set up a model (explore hyperparameter settings)
#Perceptron uses one-vs-all for multiple classes
ppnIris = Perceptron(max_iter=40,tol=0.001,eta0=1)

# Train the model
ppnIris.fit(X_train,y_train)

# Make predication
y_pred = ppnIris.predict(X_test)

# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#or use k-fold cross-validation
kf = KFold(5, shuffle=True)

#first without standardisation
print("")
print("Without standardisation")

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X,y):
    ppnIris.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = ppnIris.predict(X[validate_index])
    #print(y_test)
    #print(y_pred)
    #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1

#second with standardisation
print("")
print("With standardisation")

sc = StandardScaler()

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X,y):
    sc.fit(X[train_index])
    X_train_std = sc.transform(X[train_index])
    X_test_std = sc.transform(X[validate_index])
    ppnIris.fit(X_train_std,y[train_index])
    y_test = y[validate_index]
    y_pred = ppnIris.predict(X_test_std)
    #print(y_test)
    #print(y_pred)
    #print(f"Fold #{fold}, Training Size: {len(trainDF)}, Validation Size: {len(validateDF)}")
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1
