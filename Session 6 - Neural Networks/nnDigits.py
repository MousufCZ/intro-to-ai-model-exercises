
"""
@author: jacob
"""

from sklearn.datasets import load_digits

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.utils
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
import io
from sklearn import metrics
from sklearn.model_selection import train_test_split

#load the digits dataset from sklearn
digits = load_digits()

#set features and target 
X = digits.data
y = digits.target

print(digits.images[1])

print(digits.data.shape)

print(digits.data[1])

print(y[:20])
#use the keras built in to ensure the targets are categories
y = keras.utils.to_categorical(y)
#and check this...
print(y[:5])

#split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)

print(X_train.shape)

#set up a keras model.  
model = Sequential()
#You might try varying the activation function, and/or the number of hidden units
#model.add(Dense(128, input_dim=X.shape[1], activation='sigmoid'))
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
#you might experiment with a second hidden layer
#model.add(Dense(64, activation='relu'))
#model.add(Dense(20, activation='sigmoid'))
model.add(Dense(y.shape[1],activation='softmax'))
#compile the model setting the loss (error) measure and the optimizer
opt = keras.optimizers.SGD(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)
#model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
#Fir the model, you might change the number of epochs training is run for.
model.fit(X_train,y_train,verbose=2,epochs=256)

#make predictions (will give a probability distribution)
pred = model.predict(X_test)

print(pred[4])
#now pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))

print(pred[4])


