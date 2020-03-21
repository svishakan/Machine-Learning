# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:05:05 2019

@author: Vishakan
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
PS: EXECUTE plot statements one by one. Plotting everything at the same time is resulting in 
OVERLAP

"""


os.chdir('Datasets')

cars_data = pd.read_csv('Toyota.csv',na_values=['??','????'], index_col = 0)
cars_data.dropna(axis = 0, inplace = True)

#histogram (distplot)
sns.distplot(cars_data['Age'])
#arg should be numerical(continuous) variable
#kernel density estimate is also given

#to remove kernel density estimate (KDE)
"""
sns.distplot(cars_data['Age'],kde = True, bins = 5)
"""
#gives counts instead of KDE on y axis
#limiting no. of bins for better readability


#barplot (countplot)
#freq. dist. of categorical variables
"""
sns.countplot(x = "FuelType", data = cars_data)
"""

#grouped bar plot of FuelType and Automatic

"""
sns.countplot(x = "FuelType", data = cars_data, hue = "Automatic")
"""
print(pd.crosstab(index = cars_data['Automatic'],columns = cars_data['FuelType'],dropna = True))


