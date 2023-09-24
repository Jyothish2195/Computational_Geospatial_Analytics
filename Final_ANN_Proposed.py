# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:29:56 2021

@author: test
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
import timeit
from keras import initializers
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import numpy as np
import tensorflow as tf
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.random.seed(37)


tf.random.set_seed(89)


start = timeit.default_timer()
# load the dataset
#df = pd.read_csv('MyData_Scaled.csv')
#df = pd.read_csv('MyData_Scaled.csv')
#df = pd.read_csv('MyData_Scaled_2.csv')
df = pd.read_csv('MyData_Scaled3.csv')
#df = pd.read_csv('D.csv')

# split into input (X) and output (y) variables

X = df.iloc[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
#X = df.iloc[:,[18,24,30]]
print(X.head(0))
y = df.iloc[:,[33]]
print(y.head(0))


Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=13)


model =keras.Sequential([
    
    keras.layers.Dense(20,input_shape=(18,),activation='relu',kernel_initializer=initializers.glorot_uniform(seed=0)),
    keras.layers.Dense(7,activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
    ])

model.compile( optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
#model.fit(Xtrain,ytrain,epochs=500,verbose=1)
#history=model.fit(Xtrain,ytrain,validation_data = (Xtest,ytest), epochs=300,batch_size=8)
history=model.fit(Xtrain,ytrain,validation_data = (Xtest,ytest), epochs=158)
print(model.evaluate(Xtest,ytest))

yp=model.predict(Xtest)
print(yp[:10])

y_pred=[]
for element in yp:
    if element>0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
         

#print(y_pred)

stop = timeit.default_timer()
print(" ANN Proposed Runtime: ", stop - start)

###############  Classification Report ######################

print("  ")
print("ANN Proposed Classification_report")
print("  ")

print(classification_report(ytest,y_pred))


###############   Confusion Matrix ######################
cm= tf.math.confusion_matrix(labels=ytest,predictions=y_pred)

"""
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
"""

print(confusion_matrix(ytest, y_pred))
############## ACCURACY ################
a= accuracy_score(ytest, y_pred)*100
print("RF Accuracy Entropy",a)
###############  RMSE VALUE ######################

MSE = mean_squared_error(ytest,y_pred)
RMSE = math.sqrt(MSE)
print("RF Root Mean Square Error:",RMSE)

###############  RUNTIME ######################
stop = timeit.default_timer()
print("ANN Existing Runtime: ", stop - start)

###############  GRAPH ######################


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Proposed Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
      

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Proposed Model loss') 
plt.ylabel('Loss') 
plt.xlabel('Epoch') 
plt.legend(['Train', 'Test'], loc='upper left') 
plt.show()

import statsmodels.api as sm

lr = sm.OLS(ytrain, Xtrain).fit()
print(lr.summary())