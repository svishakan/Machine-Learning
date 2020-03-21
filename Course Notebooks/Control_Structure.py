# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
'''
=============================================================================
    Control structures
        - If elif family
        - For 
        - While 
    Functions

=============================================================================
'''
# =============================================================================
# Importing necessary libraries
# =============================================================================
import os               # ‘os’ library to change the working directory
import pandas as pd     # ‘pandas’ library to work with dataframes
import numpy as np      # ‘numpy’ library to perform numeric operations

# =============================================================================
# Importing data 
# =============================================================================
cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

# Creating copy of original data
cars_data1 = cars_data.copy()

"""
    Control Structures in Python
    - Execute certain commands only when certain condition(s) is (are) satisfied-
      (if-then-else)
    - Execute certain commands repeatedly and use a certain logic to stop
      the iteration - (for, while loops)

"""
"""
 - Creating 3 bins from the ‘Price’ variable using If Else and For Loops
 - The binned values will be stored as classes in a new column, ‘Price Class’
"""
# Inserting at end- index can only be positive

cars_data1.insert(10,"Price_Class","")

# =============================================================================
# if else and for loops
# =============================================================================

"""
    if else and for loops are implemented and the observations are separated into three categories:   
    Price 
       - up to 8450
       - between 8450 and 11950 
       - greater than 11950
    The classes have been stored in a new column ‘Price Class’
"""
import time
start = time.time()
for i in range(0,len(cars_data1['Price']),1):
    if (cars_data1['Price'][i]<=8450):
        cars_data1['Price_Class'][i]="Low"
    elif ((cars_data1['Price'][i]>11950)):
        cars_data1['Price_Class'][i]="High"
    else: cars_data1['Price_Class'][i]="Medium"

cars_data1['Price_Class'].value_counts()

end=time.time()
end-start

# =============================================================================
# while loop
# =============================================================================
"""
    - A while loop is used whenever you want to execute statements until a
      specific condition is violated
    - Here a while loop is used over the length of the column ‘Price_Class’ 
      and an if else loop is used to bin the values and store it as classes
"""
i=0
start = time.time()
while i<len(cars_data1['Price']):
    if (cars_data1['Price'][i]<=8450):
        cars_data1['Price_Class'][i]="Low"
    elif ((cars_data1['Price'][i]>11950)):
        cars_data1['Price_Class'][i]="High"
    else: cars_data1['Price_Class'][i]="Medium"
    i=i+1
    
end = time.time()
end-start

"""
    Series.value_counts() returns series containing count of unique values
"""

cars_data1['Price_Class'].value_counts()
cars_data1.insert(12,"Km_per_month",0)

# =============================================================================
# FUNCTIONS 
# =============================================================================

"""
    - A function accepts input arguments and produces an output by executing
      valid commands present in the function
    - Function name and file names need not be the same
    - A file can have one or more function definitions
    - Functions are created using
      def function_name(parameters):
       statements
    - Since statements are not demarcated explicitly, 
      it is essential to follow correct indentation practises
"""
"""
    - Converting the Age variable from months to years by defining a function
    - The converted values will be stored in a new column, ‘Age_Converted’
    - Hence, inserting a new column 
"""
cars_data1.insert(11,"Age_Converted",0)
# Here, a function c_convert has been defined
# The function takes arguments and returns one value

def c_convert(val):
    val_converted=val/12
    return val_converted

cars_data1["Age_Converted"] = c_convert(cars_data1['Age'])
cars_data1["Age_Converted"] = round(cars_data1["Age_Converted"],1)

# =============================================================================
# Function with multiple inputs and outputs
# =============================================================================

# Functions returning multiple output
# Converting months to years and getting kilometers run per month

def c_convert(val1,val2):
    val_converted=val1/12
    ratio=val2/val1
    return [val_converted,ratio]

cars_data1["Age_Converted"],cars_data1["Km_per_month"] = \
c_convert(cars_data1['Age'],cars_data1['KM'])

# =============================================================================
# END OF SCRIPT
# =============================================================================
