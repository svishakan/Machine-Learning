# -*- coding: utf-8 -*-
"""
Spyder Editor

"""
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("Datasets")

income = pd.read_csv('income.csv')

income.info()

data = income.copy()

"""
Exploratory data analysis

Getting to know the data
Data Preprocessing
Cross tables, data visualisation

"""

data.info()
#datatypes are fine

data.isnull().sum()
#no null values

data.describe()
#numerical variable description

totaldesc = data.describe(include = 'O')
#include categorical also

data['JobType'].value_counts()
data['occupation'].value_counts()
#there is spl. char ? which maybe NaN.

print(np.unique(data.JobType))
print(np.unique(data.occupation))
#finding the unique classes of data
#the spl char is ' ?'


data = pd.read_csv('income.csv', na_values = [' ?'])
#making the spl char as NaN

data.isnull().sum()
#missing data in jobtype, occupation

missing = data[data.isnull().any(axis = 1)]
#consider rows with atleast one column missing
#jobtype and occupation both are missing simultaneously
#never-worked jobtype corresponds to missing value of occupation


data2 = data.dropna(axis = 0)
#remove missing values
#because, we couldn't find relationship of interest
#or find any mechanism to fill in missing values

correlation = data2.corr()
#none of the variables are correlated with each other
#values are very less


#analysis of categorical variables
data2.columns

gender = pd.crosstab(index = data2.gender, columns = 'count', normalize = True)

print(gender)

gender_salstat = pd.crosstab(index = data2.gender, columns = data2.SalStat, margins = True, normalize = 'index')

print(gender_salstat)

salstat = sns.countplot(data2.SalStat)
#25% OF SALARY > 50000
#75% OF SALARY <= 50000

sns.distplot(data2.age, bins = 10, kde = False)
#People with age 20-45 are in high frequency

sns.boxplot('SalStat', 'age', data = data2)
data2.groupby('SalStat')['age'].median()
#People with age 35-50 are more likely to earn > 50000
#People with age 25-35 are more likely to earn <= 50000


#JobType vs. SalStat
#Education vs. SalStat
#Occupation vs. SalStat
#HoursPerWeek vs. SalStat

#Capital Gain : 92% of observations are in 0, 8% are in rest
#Capital Loss: 95% of observations are 0, 5% are in rest


#Logistic Regression
data2 = data.dropna(axis = 0)
data2.SalStat = data2.SalStat.map({' less than or equal to 50,000': 0, ' greater than 50,000': 1})    

new_data = pd.get_dummies(data2, drop_first = True)
#one hot encoding for all the categorical data to numerical data


columns_list = list(new_data.columns)
print(columns_list)

features = list(set(columns_list) - set(['SalStat']))
print(features)

y = new_data['SalStat'].values
print(y)
#separating the dependant variable SalStat from the dataset 

x = new_data[features].values
print(x)
#separating the indep. variables from the dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
#random_state = 0 is the random seed used for random sampling
#different seed => different variables will be chosen

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Model building
logistic = LogisticRegression()

logistic.fit(x_train, y_train)
logistic.coef_
logistic.intercept_


#Prediction from data
pred = logistic.predict(x_test)
print(pred)

#Confusion Matrix
#To evaluate performance of a Classification Model

conf_mat = confusion_matrix(y_test, pred)
print(conf_mat) 
#diagonals: correctly classified
#antidiagonals: incorrectly classified

#Accuracy Calc
acc = accuracy_score(y_test, pred)
print(acc)
#83.67% accurrate


print('Misclassified Samples %d' %(y_test != pred).sum())


#Removing Insignificant Variables
data2 = data.dropna(axis = 0)
data2.SalStat = data2.SalStat.map({' less than or equal to 50,000': 0, ' greater than 50,000': 1})
print(data2.SalStat)

cols = ['gender', 'nativecountry', 'race', 'JobType']
new_data = data2.drop(cols, axis = 1)

new_data = pd.get_dummies(new_data, drop_first = True)
#one hot encoding for all the categorical data to numerical data


columns_list = list(new_data.columns)
print(columns_list)

features = list(set(columns_list) - set(['SalStat']))
print(features)

y = new_data['SalStat'].values
print(y)
#separating the dependant variable SalStat from the dataset 

x = new_data[features].values
print(x)
#separating the indep. variables from the dataset


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
#random_state = 0 is the random seed used for random sampling
#different seed => different variables will be chosen

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#Model building
logistic = LogisticRegression()

logistic.fit(x_train, y_train)
logistic.coef_
logistic.intercept_


#Prediction from data
pred = logistic.predict(x_test)
print(pred)

#Confusion Matrix
#To evaluate performance of a Classification Model

conf_mat = confusion_matrix(y_test, pred)
print(conf_mat) 
#diagonals: correctly classified
#antidiagonals: incorrectly classified

#Accuracy Calc
acc = accuracy_score(y_test, pred)
print(acc)
#83.40% accurrate


print('Misclassified Samples %d' %(y_test != pred).sum())


#KNN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(x_train, y_train)

pred = knn.predict(x_test)

#Performance Metric Check

conf_mat = confusion_matrix(y_test, pred)
print(conf_mat)

#Accuracy Score
acc = accuracy_score(y_test, pred)
print(acc)
#83.38% accurate

print('Misclassified Samples %d' %(y_test != pred).sum())

misclassified_sample = []
#Calculating error for k values between 1 - 20
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    misclassified_sample.append((y_test != pred).sum())
    print('Misclassified Samples %d' %(y_test != pred).sum())
#Use k = 17, as it gives least misclassified values, as 1422
