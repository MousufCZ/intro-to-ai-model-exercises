#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 15:33:36 2020

@author: jacob
"""

#library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.svm import SVR

#load the dataset
diabetes = datasets.load_diabetes()

#print five rows of data
print(diabetes.data[:5])

#define X and y
X = diabetes.data
y = diabetes.target

#note the shape of the data
print(X.shape)

#split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

#fit a linear model to this data
linDiabetes = LinearRegression().fit(X_train, y_train)

#predict values for the testing data
y_pred = linDiabetes.predict(X_test)

#looks at some outputs side by side
df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df_head = df_compare.head(25)
print(df_head)

#some stats
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#and consider the output
print('Coefficient of determination: %.2f' % metrics.r2_score(y_test, y_pred))
print('Correlation: ', stats.pearsonr(y_test,y_pred))


#plot the outputs
#set figure size
plt.rc('figure', figsize=(8, 8))

#plot line down the middle
x = np.linspace(400,0,100)
plt.plot(x, x, '-r',color='r')

#plot the points, prediction versus actual
plt.scatter(y_test, y_pred, color='black')

plt.xticks(())
plt.yticks(())

plt.show()

#and plot the values to emphasise the noise
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
    
chart_regression(y_pred,y_test,sort=True)  


#can we get a better results with a SVM? [No]

#set hyperparameter and train a support vector machine for regression
#you need to search for an appropriate C and epsilon
svrDiabetes = SVR(kernel='linear',C=1000, epsilon=1).fit(X_train, y_train)

#produce test predictions
y_svr_pred = svrDiabetes.predict(X_test)

#evaluate
print('Mean:', np.mean(y_test))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_svr_pred)))

chart_regression(y_svr_pred,y_test,sort=True)  
