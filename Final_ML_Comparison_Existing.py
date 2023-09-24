# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:51:33 2021

@author: Jyothish
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import timeit
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn import metrics


#x=pd.read_csv("MyData_Scaled.csv")
#y=pd.read_csv("MyData_Scaled.csv")

#x=pd.read_csv("MyData_Scaled1.csv")
#y=pd.read_csv("MyData_Scaled1.csv")

x=pd.read_csv("MyData_Scaled3.csv")
y=pd.read_csv("MyData_Scaled3.csv")

#np.isnan(x.any()) #and gets False

x1=list(x["YEAR"])
y1=list(x["OUT"])

#print("SAMPLE")
#print(np.where(np.isnan(x)))
      
plt.plot(x1, y1,'*')
plt.title('Plotted YEAR to BINARY CLASSIFICATION')
plt.savefig('plot3.jpg', dpi=300, bbox_inches='tight')
plt.show()

##########################################################
print(".................. LINEAR REGRESSION ..........................")
start = timeit.default_timer()
df=pd.read_csv("MyData_Scaled3.csv")
X = df.iloc[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
X = df.iloc[:,[32]]
#X = df.iloc[:,0:32]
#print(X)
Y = df.iloc[:,[33]]
#print(Y)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

y_test=y_test.astype(int)
y_pred=y_pred.astype(int)


print("  ")
print(" Linear Regression Classification_report ")
print("  ")

print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

a= accuracy_score(y_test,y_pred)*100
print("LINEAR REGRESSION Accuracy",a)

MSE = mean_squared_error(y_test,y_pred)
RMSE = math.sqrt(MSE)
print("Linear Regression Root Mean Square Error:", RMSE)

stop = timeit.default_timer()
print("Linear Regression Runtime: ", stop - start)



import statsmodels.api as sm

lr = sm.OLS(y_train, X_train).fit()
print(lr.summary())


#######################################################
print(".................. DECISION TREE CLASSIFICATION (ENTROPY) .........................")
start = timeit.default_timer()

def importdata():
    df=pd.read_csv("MyData_Scaled3.csv")
    # Printing the dataswet shape
    #print ("Dataset Length: ", len(df))
    #print ("Dataset Shape: ", df.shape)
      
    # Printing the dataset obseravtions
    #print ("Dataset: ",df.head())
    return df

def splitdataset(df):
  
    # Separating the target variable
    #X = df.iloc[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
    X = df.iloc[:,[29]]
    Y = Y = df.iloc[:,[33]]
  
    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.3, random_state = 0)
      
    return X, Y, X_train, X_test, y_train, y_test



# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    
    print("Report : ",
    classification_report(y_test, y_pred))
      
    print("Confusion Matrix: ",confusion_matrix(y_test, y_pred))
      
    print ("DT Accuracy : ", accuracy_score(y_test,y_pred)*100)
      
    MSE = mean_squared_error(y_test,y_pred)
    RMSE = math.sqrt(MSE)
    print("DT Root Mean Square Error: ",RMSE)
    
# Driver code
def main():
      
    # Building Phase
    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    # Operational Phase
    print("Results Using Gini Index:")
      
    # Prediction using gini
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
      
    print("Results Using Entropy:")
    # Prediction using entropy
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)


# Calling main function
if __name__=="__main__":
    main()


stop = timeit.default_timer()
print("DT Runtime: ", stop - start)


#######################################################
print(".................. SVM  .........................")
start = timeit.default_timer()

df=pd.read_csv("MyData_Scaled3.csv")
X = df.iloc[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
X = df.iloc[:,[32]]
#X = df.iloc[:,0:32]
#print(X)
Y = df.iloc[:,[33]]
#print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=0.9, random_state=1)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


print("  ")
print("SVM Classification_report")
print("  ")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

a= accuracy_score(y_test,y_pred)*100
print("SVM Accuracy",a)

MSE = mean_squared_error(y_test,y_pred)
RMSE = math.sqrt(MSE)
print("SVM Root Mean Square Error:",RMSE)


stop = timeit.default_timer()
print("SVM Runtime: ", stop - start)





#######################################################
print(".................. Naive Bayes .........................")
start = timeit.default_timer()
 
df=pd.read_csv("MyData_Scaled3.csv")
#print(df.head(0))
X = df.iloc[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]]
Y = df.iloc[:,[33]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)



classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred  =  classifier.predict(X_test)

print("  ")
print("Naive Bayes  Classification_report")
print("  ")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))


#cm = confusion_matrix(y_test, y_pred)
#ac = accuracy_score(y_test,y_pred)

a= accuracy_score(y_test,y_pred)*100
print("Naive Bayes Accuracy",a)

MSE = mean_squared_error(y_test,y_pred)
RMSE = math.sqrt(MSE)
print("Naive Bayes Root Mean Square Error:",RMSE)
stop = timeit.default_timer()
print(" Naive Bayes Runtime: ", stop - start)



#######################################################

print(".................. Random Forest .........................")
start = timeit.default_timer()

df=pd.read_csv("MyData_Scaled3.csv")

X = df.iloc[:,[29]]
#X = df.iloc[:,0:32]
#print(X)
Y = df.iloc[:,[33]]
#print(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.70, random_state=5) 

clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("RF Classification_report")
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

a= accuracy_score(y_test, y_pred)*100
print("RF Accuracy Entropy",a)

MSE = mean_squared_error(y_test,y_pred)
RMSE = math.sqrt(MSE)
print("RF Root Mean Square Error:",RMSE)

stop = timeit.default_timer()
print("RF Runtime: ", stop - start)

import statsmodels.api as sm

lr = sm.OLS(y_train, X_train).fit()
print(lr.summary())