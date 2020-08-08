# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
# =============================================================================
# DATA PREPARATION
# =============================================================================

#%%
# To work with dataframes
import pandas as pd 

# To perform numerical operations
import numpy as np

#%%
# Importing data 

demoDetails    =  pd.read_csv("demoDetails.csv"   , index_col=0)
acDetails      =  pd.read_csv("acDetails.txt"     , sep="\t")
serviceDetails =  pd.read_csv("serviceDetails.csv", index_col=0)

# By setting 'index_col = 0', 1st column will be the index column

#%%
# Data Wrangling
"""
 - We are interested in merging acDetails, demoDetails and serviceDetails 
 - Before merging we need to make necessary checks !
 - What are the mandatory checks you should look for before merging ?
 -   1. Are there any duplicate records?
     2. Whether the customer ID is common across all the files ?
"""

# 1. Are there any duplicate records?

len(np.unique(demoDetails['customerID']))

len(np.unique(acDetails['customerID']))

len(np.unique(serviceDetails['customerID']))

# Yes, there is one duplicate record across all the three dataframes

#%%
# ======================== Determining duplicate records =================================

# To determine the duplicate records 'duplicated()' can be used

demoDetails.duplicated(subset=['customerID'], keep=False)

# duplicated function returns a Boolean Series with True value 
# for each duplicated row

# So now let's subset the rows and look at the duplications

demoDetails[demoDetails.duplicated(['customerID'],keep=False)]

acDetails[acDetails.duplicated(['customerID'],keep=False)]

serviceDetails[serviceDetails.duplicated(['customerID'],keep=False)]

#%%
# ====================== Removing duplicate records ================================

demoDetails    =  demoDetails.drop_duplicates() 

acDetails      =  acDetails.drop_duplicates()

serviceDetails =  serviceDetails.drop_duplicates()

# First occurrence of the duplicate row is kept and 
# subsequent occurrence have been removed

#%%

# 2. Whether the customer ID is common across all the files ?

# syntax: dataframe1.equals(dataframe2)

acDetails.customerID.equals(demoDetails.customerID)

serviceDetails.customerID.equals (demoDetails.customerID)

acDetails.customerID.equals (serviceDetails.customerID)

# Looks like they are indeed identical!

#%% 
# ====================== Joining three dataframes =============================

# Syntax: pd.merge(df1, df2, on=['Column_Name'], how='inner')

churn  =  pd.merge(demoDetails, acDetails, on = "customerID")

churn  =  pd.merge(churn,serviceDetails,   on = "customerID")

churn1 =  churn.copy()

#%%
# ============ Data Exploration / Understanding the data ======================

churn1.info()

""" Points to note:
-'tenure' has been read as object instead of integer
-'SeniorCitizen' has been read as float64 instead of object
- Missing values present in few variables
"""
# unique() finds the unique elements of an array
np.unique(churn1['tenure'], return_counts = True )

# 'tenure' has been read as object instead of integer 
# because of values One/Four which are strings

np.unique(churn1['SeniorCitizen'])

# 'SeniorCitizen' has been read as float64 instead of int64 since it has values nan values

# Checking frequencies of each categories in a variable

categotical_data = churn1.select_dtypes(include=['object']).copy()

categotical_data.columns

categotical_data['gender'].value_counts() 

# categotical_data.value_counts() AttributeError:

categotical_data = categotical_data.drop(['customerID','tenure'],axis = 1)

frequencies      = categotical_data.apply(lambda x: x.value_counts()).T.stack()

print(frequencies)

""" Points to note:
- 'Dependents' should have only 2 levels (Yes/No) but it has 3 due 
-  the special character '1@#' that has been read as another level
"""
# Summary of numerical variables

summary = churn1.describe()

print(summary)

#%%
# ======================================== Data Cleaning ======================

# Cleaning column 'tenure'

# Replacing 'Four' by 4 and 'One' by 1 in 'tenure'
    
churn1['tenure'] = churn1.tenure.replace("Four", 4)

churn1['tenure'] = churn1.tenure.replace("One", 1) 

churn1['tenure'] = churn1.tenure.astype(int)

print(churn1['tenure'])

###############################################################################

# Cleaning column 'Dependents'
""" 'Dependents' should have only 2 levels (Yes/No) but it has 3 due 
     the special character '1@#' that has been read as another level"""
     
# Gives counts- class 'No' has the max count 

pd.crosstab(index=churn1['Dependents'], columns="count")

# Replacing "1@#" with 'No'   

churn1['Dependents'] = churn1['Dependents'].replace("1@#", 'No')
      
# Verifying if the special characters were converted to desired class

table_dependents  = pd.crosstab(index = churn1['Dependents'], columns="count")

print(table_dependents) 

#%%
"""
In this lecture:
    - Checking for logical fallacies in the data
    - Approaches to resolve the logical fallacies in the data
    - Outlier detection using boxplot
    - Approaches to fill in missing values
    - Simple random sampling
"""
############################### Logical Checks ################################
# 1. Checking if the 'customerID' is consistent

print(churn1['customerID'])

"""


I  Interms of total number of characters
II Sequence of charaters i.e. first 4 characters of customerID should be 
    numbers followed by hyphen and 5 upper case letters
    
"""
# I
# to get the index of customerID whose length != 10
len_ind = [i for i,value in enumerate(churn1.customerID) if len(value)!=10]

import re 
pattern = '^[0-9]{4,4}-[A-Z]{5,5}'   
p = re.compile(pattern)
type(p)

q = [i for i,value in enumerate(churn1.customerID) if p.match(str(value))==None]
print(q)

fp1 = re.compile('^[A-Z]{5,5}-[0-9]{4,4}')
fp2 = re.compile('^[0-9]{4,4}/[A-Z]{5,5}')

for i in q:
    false_str = str(churn1.customerID[i])
    if(fp1.match(false_str)):
        str_splits=false_str.split('-')
        churn1.customerID[i]=str_splits[1]+'-'+str_splits[0]
    elif(fp2.match(false_str)):
        churn1.customerID[i]=false_str.replace('/','-')

#%%
#################################################################################
# Logical checks - check for fallacies in the data
# If Internet service = No, then all the allied services related to internet 
# should be no. 
        
# Is that the case?

# Subsetting Internet Service and allied services
y = churn1[(churn1.InternetService =='No')]
z = y.iloc[:,13:20]

"""
   Some observations have InterService= No and Yes in certain allied services
   This is a logical fallacy!
   Two ways of approach:
   => Brute force method- wherever InternetService = No, blindly make other 
      related fields 'No'
   => Logical approach- If there are say 2 or more Yes in the allied services,
      then go back and change InternetService= Yes 
                       else change the allied services = No
"""
# Logical approach

for i,row in z.iterrows():
    yes_cnt=row.str.count('Yes').sum()
    if(yes_cnt>=2):
        z.loc[i].InternetService='Yes'
    else:
        z.loc[i,:]='No internet service'


###############################################################################
# OUTLIER DETECTION
###############################################################################

## looking for any outliers
churn1.tenure.describe()

# Outlier detection using boxplot

import seaborn as sns

sns.boxplot(y = churn1['tenure'])

# Replacing outliers by median of column 'tenure'
churn1['tenure'] = np.where(churn1['tenure']>=500,
      churn1['tenure'].median(), churn1['tenure'])

# Checking the summary of the column 'tenure’ after median imputation
churn1['tenure'].describe()
sns.boxplot(y = churn1['tenure'])

# =============================================================================
# Identifying missing values
# =============================================================================
# To check the count of missing values present in each column 
churn1.isnull().sum()

# Missing values in SeniorCitizen, MonthlyCharges, TotalCharges
# =============================================================================
# Imputing missing values
# =============================================================================
""" Two ways of approach
	 - Fill the missing values by mean / median, in case of numerical variable
	 - Fill the missing values with the class which has maximum count, in case of
       categorical variable
"""

# ==================== Mode imputation - SeniorCitizen ========================

churn1['SeniorCitizen'].fillna(churn1['SeniorCitizen'].mode()[0], inplace = True)

churn1.SeniorCitizen.isnull().sum()

###############################################################################
# Look at the description to know whether numerical variables should be 
# imputed with mean or median
"""
    DataFrame.describe() - generates descriptive statistics that summarize the 
    central tendency, dispersion and shape of a dataset’s distribution,
    excluding NaN values
"""
churn1.describe()
# ==================== Mean imputation - TotalCharges ========================

churn1['TotalCharges'].mean()

sns.boxplot(x = churn1['TotalCharges'], y = churn1['Churn'])

# Let us impute those missing values using mean based on the output
# varieble 'Churn' – Yes & No

churn1.groupby(['Churn']).mean().groupby('Churn')['TotalCharges'].mean()

churn1['TotalCharges'] = churn1.groupby('Churn')['TotalCharges']\
.transform(lambda x: x.fillna(x.mean()))

churn1.TotalCharges.isnull().sum()


# ==================== Mean imputation - MonthlyCharges ========================

churn1['MonthlyCharges'].mean()

sns.boxplot(x = churn1['MonthlyCharges'], y = churn1['Churn'])

# Let us impute those missing values using mean based on the output
# varieble 'Churn' – Yes & No

churn1.groupby(['Churn']).mean().groupby('Churn')['MonthlyCharges'].mean()

churn1['MonthlyCharges'] = churn1.groupby('Churn')['MonthlyCharges']\
.transform(lambda x: x.fillna(x.mean()))

churn1.MonthlyCharges.isnull().sum()

###############################################################################
# SAMPLING
###############################################################################

# =================== RANDOM SAMPLING -  WITH REPLACEMENT ==================
 
import random

p1    = list(range(1, 20))
print(p1)

SRSWR = random.sample(population = p1, k = 10)
print(SRSWR)

# If the sample size i.e. k is larger than the popultaion p1, ValueError is raised.

# =================== RANDOM SAMPLING -  WITHOUT REPLACEMENT ==================

p2 = list(range(1, 25))
print(p2)

SRSWOR = random.choices(population = p2, k = 10)
print(SRSWOR)

###############################################################################
############################## MODULE OUTCOMES ################################
###############################################################################
#1. Importing from different formats                                          
#2. Joins in python                                                           
#3. Basic descriptive analysis of data - to check the data type               
#4. Convert to valid data types                                               
#5. Consistency checks, unique values and regular expression patterns         
#6. Logical checks for outliers                                               
#7. Filling missing data - avg of all data, avg of data in categories,        
## apply lambda                                                               
#8. Outlier detection                                                         
#9. Sampling methods - random (w.o.r,w.r), stratified                         
###############################################################################

###############################################################################
############################### END OF SCRIPT #################################
###############################################################################
