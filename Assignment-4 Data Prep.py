# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:28:15 2019

@author: svish

Data Preparation Assignment
"""


import numpy as np
import pandas as pd
import seaborn as sns
import re
import os

os.chdir("Datasets")

file1 = pd.read_csv("stockfile1.csv")
file2 = pd.read_csv("stockfile2.txt")

stocks = pd.concat([file1,file2], ignore_index = True)
#merging both files by the row

stocks.info()
#Volume is of object type, not int64. need to check

#dates are unique
date_uniq = len(np.unique(stocks.Date))
print(date_uniq)

print(stocks.Volume.value_counts())
#need to replace zero by 0

stocks1 = stocks.copy()

stocks1.Volume = stocks1.Volume.replace("zero",0)
stocks1.Volume = stocks1.Volume.astype(int)
stocks1.info()

print(stocks1.isnull().sum())

stocks1.describe()

#checking for logical fallacies

#at vol = 0, if all 3 values are equal.
"""
The vol = 0 corresponding values:(Open Low High Close)
(458.75, 458.75, 458.75, 458.75)
(1447.35, 487.35, 487.35, 487.35)
(1755.8, 795.8, 795.8, 795.8)
(1727.85, 767.85, 767.85, 767.85)
(767.25, 767.25, 767.25, 767.25)
(1778.0, 818.0, 818.0, 818.0)
(519.55, 519.55, 519.55, 519.55)
(1800.45, 840.45, 840.45, 840.45)
(2291.95, 1331.95, 1331.95, 1331.95)
(1237.75, 1237.75, 1237.75, 1237.75)
(1201.7, 1201.7, 1201.7, 1201.7)
"""

for i, row in stocks1.iterrows():
    if row.Volume == 0:
        if(row.Open != row.Close):
            stocks1.Open[i] = stocks1.Close[i]
            
#anomaly fixed

#check and fix if High and Low are appropriate i.e. High >= Low

for i,row in stocks1.iterrows():
    if row.High < row.Low:
        stocks1.High[i], stocks1.Low[i] = stocks1.Low[i], stocks1.High[i]

print(stocks1.describe())
#open mean and median OK
#high mean and median OK
#low mean and median OK
#close mean and median OK
#volume mean and median NOT OK

sns.boxplot(y = stocks1.Volume)
stocks1.Volume = np.where(stocks1.Volume>70000, stocks1.Volume.median(), stocks1.Volume)
sns.boxplot(y = stocks1.Volume)

print(stocks1.describe())

stocks1.isnull().sum()


#filling missing values:
#open missing or close missing


for i, row in stocks1.iterrows():
    if(np.isnan(row.Open)):
        stocks1.Open[i] = round((row.Close + row.High + row.Low)/3, 2)
    if(np.isnan(row.Close)):
        stocks1.Close[i] = round((row.Open + row.High + row.Low)/3, 2)

stocks1.isnull().sum()


#high or low missing

for i, row in stocks1.iterrows():
    if(np.isnan(row.High)):
        stocks1.High[i] = max([row.Open, row.Close])
    if(np.isnan(row.Low)):
        stocks1.Low[i] = min([row.Open, row.Close])
        
stocks1.isnull().sum()


stocks1.describe()

sns.pairplot(data = stocks1)
sns.distplot(stocks1.Volume)


anskey = pd.read_csv("stockdata_answerkey.csv")
sns.distplot(anskey.Volume)


stocks1.to_csv("stocksdata_myanswer.csv")
