# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
'''
=============================================================================
 Pandas Dataframes
    - Introduction to pandas
    - Importing data into Spyder
    - Creating copy of original data
    - Attributes of data
    - Indexing and selecting data
    - Data types :Numeric & Character
    - Checking data types of each column
    - Count of unique data types
    - Selecting data based on data types
    - Concise summary of dataframe
    - Checking format of each column
    - Getting unique elements of each columns
    - Converting variable’s data types
    - Category vs Object data type
    - Cleaning column ‘Doors
    - Getting count of missing values
    - Frequency tables
    - Two-way tables
    - Two-way table - joint probability
    - Two-way table - marginal probability
    - Two-way table - conditional probability
    - Correlation
    - Identifying missing values
    - Approaches to fill the missing values
=============================================================================
'''
# =============================================================================
# Importing necessary libraries
# =============================================================================
import os               # ‘os’ library to change the working directory
import pandas as pd     # ‘pandas’ library to work with dataframes
import numpy as np      # ‘numpy’ library to perform numeric operations

#os. chdir("D:\Pandas") # Changing the working directory

# =============================================================================
# Importing data 
# =============================================================================
cars_data = pd.read_csv('Toyota.csv')

# By passing 'index_col=0', first column becomes the index column
cars_data = pd.read_csv('Toyota.csv', index_col=0)

# =============================================================================
# Creating copy of original data
# =============================================================================
'''
In Python, there are two ways to create copies
 * Shallow copy :
 - It only creates a new variable that shares the reference of the original object
 - Any changes made to a copy of object will be reflected in the original object as well
 * Deep copy: 
 - In case of deep copy, a copy of object is copied in other object with no 
   reference to the original
 - Any changes made to a copy of object will not be reflected in the original object
'''
# shallow copy  
samp = cars_data                     
samp = cars_data.copy(deep=False)   

# deep copy 
cars_data1 = cars_data.copy()      
cars_data1 = cars_data.copy(deep=True)

# =============================================================================
# Attributes of data
# =============================================================================
cars_data1.index         # to get the index (row labels) of the dataframe
cars_data1.columns       # to get the column labels of the dataframe
cars_data1.size          # to get the total number of elements from the dataframe
cars_data1.shape         # to get the dimensionality of the dataframe
cars_data1.memory_usage()# to get the memory usage of each column in bytes
cars_data1.ndim          # to get the number of axes / array dimensions
                         # a two-dimensional array stores data in a format
                         # consisting of rows and columns

# =============================================================================
# Indexing and selecting data
# =============================================================================
"""
 - Python slicing operator ‘[ ]’ and attribute/ dot operator ‘. ’  are used 
   for indexing
 - Provides quick and easy access to pandas data structures
"""
cars_data1.head(6) # The function head returns the first n rows from the dataframe
                   # By default, the head() returns first 5 rows
                  
cars_data1.tail(5) # The function tail returns the last n rows 
                   # for the object based on position

"""
 -  To access a scalar value, the fastest way is to use the 'at' and 'iat' methods
 - 'at' provides label-based scalar lookups
 - 'iat' provides integer-based lookups 
"""
cars_data1.at[4,'FuelType'] # value corresponds to 5th row & 'FuelType' column
cars_data1.iat[5,6]         # value corresponds to 6th row & 7th column

"""
    To access a group of rows and columns by label(s) .loc[ ] can be used
"""
cars_data1.loc[:,'FuelType'] 
# =============================================================================
# Data types
# =============================================================================
"""
    - The way information gets stored in a dataframe or a python object affects
      the analysis and outputs of calculations
    - There are two main types of data : numeric and character types
    - Numeric data types includes integers and floats
    - For example: integer – 10, float – 10.53    
    - Strings are known as objects in pandas which can store values that contain
    - numbers and / or characters
    - For example: ‘category1’
"""
# =============================================================================
# Checking data types of each column
# =============================================================================
cars_data1.dtypes             # returns a series with the data type of each column

# =============================================================================
# Count of unique data types
# =============================================================================
cars_data1.get_dtype_counts() # returns counts of unique data types in the dataframe

# =============================================================================
# Selecting data based on data types
# =============================================================================
cars_data1.select_dtypes(exclude=[object])
# returns a subset of the columns from dataframe by excluding columns of object data 

# =============================================================================
# Concise summary of dataframe
# =============================================================================
"""
info() returns a concise summary of a dataframe
    data type of index
    data type of columns
    count of non-null values 
    memory usage
"""
cars_data1.info()
# =============================================================================
# Checking format of each column
# =============================================================================
"""
By using info(), we can see
    - ‘KM’ has been read as object instead of integer
    - ‘HP’ has been read as object instead of integer
    - ‘MetColor’ and ‘Automatic’ have been read as float64 and int64 respectively
       since it has values 0/1
    - Ideally, ‘Doors’ should’ve been read as int64 since it has values 2, 3, 4, 5.
      But it has been read as object
    - Missing values present in few variables
Let’s encounter the reason !
"""
# =============================================================================
# Unique elements of columns
# =============================================================================
""" unique() is used to find the unique elements of a column """

print(np.unique(cars_data1['KM'])) # ‘KM’ has special character to it '??'  
                                   # Hence, it has been read as object instead of int64
                                   
print(np.unique(cars_data1['HP'])) # ‘HP’ has special character to it '????'   
                                   # Hence, it has been read as object instead of int64
                                   
print(np.unique(cars_data1['MetColor'])) # ‘MetColor’ have been read as float64
                                         #  since it has values 0. & 1.

print(np.unique(cars_data1['Automatic']))# ‘Automatic’ has been read as int64
                                         #  since it has values 0 & 1
                                         
print(np.unique(cars_data1['Doors']))    # ‘Doors’ has been read as object 
                                         # instead of int64 because of values 
                                         # ‘five’ ‘four’ ‘three’ which are strings

# =============================================================================
# Importing data (replacing special chars with nan values)
# =============================================================================
"""
    - We need to know how missing values are represented in the dataset
      in order to make reasonable decisions 
    - The missing values exist in the form of ‘nan’, '??', '????'
    - Python, by default replace blank values with ‘nan’
    - Now, importing the data considering other forms of missing values in a dataframe
"""
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

cars_data.info() # Summary - after replacing special characters with nan

# =============================================================================
# Converting variable’s data types
# =============================================================================
""" astype() method is used to explicitly convert data types from one to another"""

# Converting ‘MetColor’ , ‘Automatic’ to object data type

cars_data['MetColor']  = cars_data['MetColor'].astype('object')
cars_data['Automatic'] = cars_data['Automatic'].astype('object')

# =============================================================================
# category vs object data type
# =============================================================================
""" nbytes() is used to get the total bytes consumed by the elements of the columns"""

# If ‘FuelType’ is of object data type,
cars_data['FuelType'].nbytes                     # 11488                 

# If ‘FuelType’ is of category data type,
cars_data['FuelType'].astype('category').nbytes # 1460

# Re-checking the data type of variables after all the conversions
cars_data.info()

# =============================================================================
# Cleaning column ‘Doors’
# =============================================================================
# Checking unique values of variable ‘Doors’ :
print(np.unique(cars_data['Doors']))

"""
    replace() is used to replace a value with the desired value 
    Syntax: DataFrame.replace([to_replace, value, …])
"""

cars_data['Doors'].replace('three',3,inplace=True)
cars_data['Doors'].replace('four',4,inplace=True)
cars_data['Doors'].replace('five',5,inplace=True)

# To check the frequencies of unique cateogories in a variable
cars_data['Doors'].value_counts()

"""
   (or) Pandas- where() 
"""
cars_data['Doors'].where(cars_data['Doors']!='three',3,inplace=True)

""" 
   (or) Numpy- where()
"""
cars_data['Doors'] = np.where(cars_data['Doors']=='five',5,cars_data['Doors'])

# Converting ‘Doors’ to int64:
cars_data['Doors'] = cars_data['Doors'].astype('int64')
cars_data['Doors'].value_counts()

# =============================================================================
# To detect missing values
# =============================================================================
# To check the count of missing values present in each column Dataframe.isnull.sum() is used

cars_data.isnull().sum()

# =============================================================================
#   Cross tables & Correlation
# =============================================================================
cars_data2 = cars_data.copy()
"""
    pandas.crosstab()
    - To compute a simple cross-tabulation of one, two (or more) factors
    - By default computes a frequency table of the factors 
"""
# =============================================================================
#     # One way table    
# =============================================================================

pd.crosstab(index=cars_data2['FuelType'], columns='count', dropna=True)
# Most of the cars have petrol as fuel type

# =============================================================================
#     # Two-way table 
# =============================================================================
# To look at the frequency distribution of gearbox types with respect to different
# fuel types of the cars

pd.crosstab(index   = cars_data2['Automatic'], 
            columns = cars_data2['FuelType'],
            dropna  = True)

# =============================================================================
#     # Two-way table with proportion / Joint probability
# =============================================================================
"""
Joint probability is the likelihood of two independent events happening at the same time
"""
pd.crosstab(index     = cars_data2['Automatic'], 
            columns   = cars_data2['FuelType'],
            normalize = True,
            dropna    = True)

# 0.82 => Joint probability of manual gear box and petrol fuel type

# =============================================================================
#     Two-way table - Marginal probability
# =============================================================================
"""
Marginal probability is the probability of the occurrence of the single event
"""
pd.crosstab(index     = cars_data2['Automatic'], 
            columns   = cars_data2['FuelType'],
            margins   = True,
            dropna    = True,
            normalize = True)

# Probability of cars having manual gear box when the fuel type are
# CNG or Diesel or Petrol is 0.95

# =============================================================================
#     Two-way table - Conditional probability=> Row sum = 1
# =============================================================================
"""
Conditional probability is the probability of an event ( A ), given that 
another event ( B ) has already occurred
"""
pd.crosstab(index     = cars_data2['Automatic'], 
            columns   = cars_data2['FuelType'],
            margins   = True,
            dropna    = True,
            normalize = 'index')

# Given the gear box, probability of different fuel type

# =============================================================================
#     Two-way table - Conditional probability => Column sum =1
# =============================================================================
pd.crosstab(index     = cars_data2['Automatic'], 
            columns   = cars_data2['FuelType'],
            margins   = True,
            dropna    = True,
            normalize = 'columns')

# Given the fuel type, probability of different gear box 
   
# =============================================================================
# Correlation    
# =============================================================================
# Correlation: the strength of association between two variables 

# Excluding the categorical variables to find the correlation

numerical_data = cars_data2.select_dtypes(exclude=[object])
print(numerical_data.shape)

# Finding the correlation between numerical variables
corr_matrix = numerical_data.corr()
print(corr_matrix)

# Rounding off to two decimal places
print(round(corr_matrix,2))

# =============================================================================
# Identifying missing values
# =============================================================================
"""
 - In Pandas dataframes, missing data is represented by NaN
  (an acronym for Not a Number)
 - To check null values in Pandas dataframes, isnull() and isna() are used
 - These functions returns a dataframe of Boolean values which are True for NaN values
"""
cars_data2 = cars_data.copy()
cars_data3 = cars_data2.copy()

# To check the count of missing values present in each column 

print('Data columns with null values:\n')

cars_data2.isna().sum()    #or
cars_data2.isnull().sum()

# Subsetting the rows that have one or more missing values
missing = cars_data2[cars_data2.isnull().any(axis=1)]

# =============================================================================
# Imputing missing values
# =============================================================================
""" Two ways of approach
	 - Fill the missing values by mean / median, in case of numerical variable
	 - Fill the missing values with the class which has maximum count, in case of
       categorical variable
"""

# Look at the description to know whether numerical variables should be 
# imputed with mean or median
"""
    DataFrame.describe() - generates descriptive statistics that summarize the 
    central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values
"""
cars_data2.describe()
cars_data2.describe(include="O")
cars_data2.describe(include="all")


# Mean and median of kilometer is far away
# Therefore impute with median

# ==================== Replacing 'Age' with mean ==============================
cars_data2['Age'].mean()
 
cars_data2['Age'].fillna(cars_data2['Age'].mean(), inplace = True)

cars_data2['Age'].isnull().sum()

# ==================== Replacing 'KM' with median ==============================
cars_data2['KM'].median()

cars_data2['KM'].fillna(cars_data2['KM'].median(), inplace = True)

cars_data2['KM'].isnull().sum()

# ==================== Replacing 'HP' with mean ==============================
cars_data2['HP'].mean()

cars_data2['HP'].fillna(cars_data2['HP'].mean(), inplace = True)

cars_data2['HP'].isnull().sum()

# Check for missing data after filling values
cars_data2.isnull().sum()

# ==================== Replacing 'Fuel Type' with mode ========================
"""
- Returns a Series containing counts of unique values
- The values will be in descending order so that the first element is 
  the most frequently-occurring element
- Excludes NA values by default
"""
cars_data2['FuelType'].value_counts() 

# To get the mode value of FuelType
cars_data2['FuelType'].value_counts().index[0]

# To fill NA/NaN values using the specified value
cars_data2['FuelType'].fillna(cars_data2['FuelType']\
      .value_counts().index[0],\
      inplace = True)

cars_data2['FuelType'].isnull().sum()

# ==================== Replacing 'MetColor' with mode ========================

# To get the mode value of Metcolor
cars_data2['MetColor'].mode()

# To get categroy with maximum freq
# Index 0 will get the category
cars_data2['MetColor'].mode()[0]

# replacing MetColor with mode
cars_data2['MetColor'].fillna(cars_data2['MetColor']\
      .mode()[0], inplace = True)

## Check for missing data after filling values 
cars_data2['MetColor'].isnull().sum()

# Check for missing data after filling values
cars_data2.isnull().sum()

# ==================== Imputation using lambda functionss ========================

# To fill the NA/ NaN values in both numerical and categorial variables at one stretch

cars_data3 = cars_data3.apply(lambda x:x.fillna(x.value_counts().index[0]))
cars_data3.isnull().sum()

# Fill all numerical variables at a stretch
cars_data3 = cars_data3.apply(lambda x:x.fillna(x.mean()))
print('Data columns with null values:\n', cars_data3.isnull().sum())

# Fill numerical and categorial variables at one stretch

cars_data3 = cars_data3.apply(lambda x:x.fillna(x.mean()) \
                          if x.dtype=='float' else \
                          x.fillna(x.value_counts().index[0]))

print('Data columns with null values:\n', cars_data3.isnull().sum())

# =============================================================================
# END OF SCRIPT
# =============================================================================
