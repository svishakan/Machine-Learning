# -*- coding: utf-8 -*-
"""
Data Preparation 
on 
Churning of Telecom Company
(i.e. switching Network Service Provider from the specified Company)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir("Datasets")

demo = pd.read_csv("demoDetails.csv", index_col = 0)
ac = pd.read_csv("acDetails.txt", sep = "\t")
service = pd.read_csv("serviceDetails.csv", index_col = 0)

#merging ac, demo and service
#check for duplicates, cust_ID commonality
cust_unique_demo = len(np.unique(demo['customerID']))
cust_unique_ac = len(np.unique(ac['customerID']))
cust_unique_service = len(np.unique(service['customerID']))
print(cust_unique_demo, cust_unique_ac, cust_unique_service)

#there is one duplicated customerID in all 3 tables.
#As there are 251 columns.

#the duplicate is found using:
#returns boolean of each unique customerID
#keep takes "first", "last", "False"
#first occurrence is treated as unique of duplicated recs.
#last occurrence is treated as unique of duplicated recs.
#False takes all occurrences of the same rec. value as duplicate
demo.duplicated(subset = ['customerID'], keep = False)


#gives only the subset of the table where it satisfies the condition inside i.e. duplication
demo[demo.duplicated(subset = ['customerID'], keep = False)]
ac[ac.duplicated(subset = ['customerID'], keep = False)]
service[service.duplicated(subset = ['customerID'], keep = False)]

#all 3 have the same duplicate record at 45 and 250 


#remove duplicates
demo = demo.drop_duplicates()
ac = ac.drop_duplicates()
service = service.drop_duplicates()

#FIRST OCCURRENCE IS KEPT AND THE OTHER OCCCURRENCES ARE DROPPED
#Here, 45th record is kept.

#whether cust_ID is common across all files?
# using dataframe1.equals(dataframe2)
ac.customerID.equals(demo.customerID)
service.customerID.equals(service.customerID)
ac.customerID.equals(service.customerID)

#all three have same customer ID

#merging all the dataframes using custID

#how parameter of merge is by default "inner" which is intersection
#of both the tables
churn = pd.merge(demo,ac, on = "customerID")
churn = pd.merge(churn, service, on = "customerID")

churn1 = churn.copy()



#Data Cleaning

churn1.info()

#Since Tenure has float64 type
np.unique(churn1['tenure'], return_counts = True)
#return_counts returns count of each type of value
#it has string values "Four" and "One"

np.unique(churn1['SeniorCitizen'])
#has nan values 
#that is why it has been read as FLOAT64 instead of INT64

#studying categorical type data

categorical_data = churn1.select_dtypes(include=['object']).copy()

categorical_data.columns


categorical_data['gender'].value_counts()
#returns count of unique values
#value_counts() can only be applied on a column(series), not dataframe

#thus we simplify as:

categorical_data = categorical_data.drop(['customerID', 'tenure'], axis = 1)
#since we do not need these columns, as they don't provide much info
#custID is unique
#axis = 1 to drop row - wise

freq = categorical_data.apply(lambda x: x.value_counts()).T.stack()
#apply takes an argument to apply function to the dataframe
#by default axis = 0, which is column - wise
#.T.stack() is to stack the data in tabular format
print(freq)


#dependents has Yes or No which is correct
#but 1@# is a special char which should not exist.

summary = churn1.describe()
#describes the statistic of the numerical data
#categorical data are excluded.
#excludes nan values by default
print(summary)

#senior citizen is a 0/1 type data so it is meaningless to look at its' statistic of mean, median etc.


#replacing "Four" by 4 and "one" by 1 in tenure column

churn1['tenure'] = churn1.tenure.replace("Four",4)
churn1['tenure'] = churn1.tenure.replace("One",1)
churn1['tenure'] = churn1.tenure.astype(int)
print(churn1['tenure'])


#cleaning dependents column

#replace 1@#

#check proportions of each Yes-No

pd.crosstab(index = churn1['Dependents'], columns = "count")

#index arg and columns = count because we need count of each category in the column dependents.

#Since No >> Yes, we can replace 1@# by No

churn1['Dependents'] = churn1['Dependents'].replace("1@#","No")

table_dependents = pd.crosstab(index = churn1['Dependents'], columns = "count")
print(table_dependents)

#cleaned.


"""
Now check for Logical Fallacies
Attempt to solve the fallacies
Outliers detection using Box plot
Fill Missing Values
Random Sampling
"""

#checking for consistency of custID
#10 char: First 4 num, hyphen, 5 uppercase letters.
print(churn1.customerID)

len_ind = [i for i, value in enumerate(churn1.customerID) if len(value)!=10]
#no elements. therefore all are of length 10

#checking the sequence

import re #regular expressions

pattern = '^[0-9]{4,4}-[A-Z]{5,5}'

p = re.compile(pattern)
#convert to regexp pattern

type(p)

q = [i for i, value in enumerate(churn1.customerID) if p.match(str(value))==None]
#if the ID doesn't match, the match function returns None. If match, returns the object.
#gets the indices of the not matching custID rows
print(q)


#by looking at the Dataframe, we get 2 cases of falsepatterns
fp1 = re.compile('^[A-Z]{5,5}-[0-9]{4,4}')
#falsepattern1

fp2 = re.compile('^[0-9]{4,4}/[A-Z]{5,5}')
#falsepattern2

for i in q:
    false_str = str(churn1.customerID[i])
    if(fp1.match(false_str)):
        str_splits = false_str.split('-')  #splitting string into a list at - and swapping their position
        churn1.customerID[i] = str_splits[1]+ '-' +str_splits[0]
    elif(fp2.match(false_str)):
        churn1.customerID[i] = false_str.replace('/','-')
    else:
        continue



#Fallacy Checking
#if there is no internet service, then internet services like online backup, online tv streaming should also be no

y = churn1[(churn1.InternetService == "No")]
z = y.iloc[:,13:20] #since we need only columns 13-20 with internet related info


#Two ways to approach:
"""
 (brute force)
 if internet service is no, then make all allied service no
 (logical approach)
 if there are 2 allied service or more as yes, then change internet service as no
 
"""
 
 
 #logical approach is taken
 
for i, row in z.iterrows():
    yes_count = row.str.count("Yes").sum()
    if(yes_count >= 2):
        z.loc[i].InternetService = "Yes"
    else:
        z.loc[i,:] = "No internet service"
        
    

#merging the changed values with the new values
churn3 = churn.copy()
for i, row in z.iterrows():
    churn3.iloc[i,13:20] = z.loc[i]
    



#outlier detection
churn1.tenure.describe()
#mean and median aren't close
#therefore boxplot
sns.boxplot(y = churn1['tenure'])

#imputing
#replacing the tenure if it is greater than 500 with the median(since mean is affected by outlier) otherwise keep it as it is
churn1['tenure'] = np.where(churn1['tenure']>=500, churn1['tenure'].median(), churn1['tenure'])

sns.boxplot(y = churn1['tenure'])


#identify missing values

churn1.isnull().sum()
#imputing missing values
churn1['SeniorCitizen'].fillna(churn1['SeniorCitizen'].mode()[0], inplace = True)
#replace on churn1 itself using inplace = True

churn1.SeniorCitizen.isnull().sum()


churn1.describe()

churn1['TotalCharges'].mean()

sns.boxplot(x = churn1['TotalCharges'], y = churn1["Churn"])
#we have to impute based on group's mean since the 2 groups, churned and not churned, have
#differing mean values

churn1.groupby("Churn")['TotalCharges'].mean()

#filling the missing values of TotalCharges column with their respective group's mean
churn1['TotalCharges'] = churn1.groupby('Churn')['TotalCharges'].transform(lambda x: x.fillna(x.mean()))

churn1.TotalCharges.isnull().sum()


churn1.groupby(['Churn'])['MonthlyCharges'].mean()

churn1['MonthlyCharges'] = churn1.groupby(['Churn'])['MonthlyCharges'].transform(lambda x: x.fillna(x.mean()))

churn1.MonthlyCharges.isnull().sum()


#Random Sampling
#Two methods: With Replacement and Without Replacement

#with replacement: replaces repeating values of the sample with another value chosen from the population
#without replacement: doesn't replace

#example:
import random
pop = list(range(1,20))
RSWOR = random.choices(population = pop, k = 10) #population, sample size(k) k<=size(population)
#without replacement

RSWR = random.sample(population = pop, k = 10)
#with replacement

