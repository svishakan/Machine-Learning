# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:06:46 2019

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

#box and whiskers plot
#intepret the five number summary
#min max mean median mode

#sns.boxplot(y = cars_data['Price'])

#horizontal : box
#vertical : whisker
#lower whiskers and upper whiskers: below and above the box as a horizontal line
#whiskers excludes the outliers
#points above and below the whiskers are outliers

#boxplot for numerical vs categorical 
#price vs fuel types

#sns.boxplot(x = cars_data['FuelType'], y = cars_data['Price'])

#median value is greatest for the petrol type
#extreme values are found in diesel
#outliers are in diesel and petrol
#diesel has the highest max. and min cost. (Horizontal Whisker location)

#grouped box and whiskers plot

#sns.boxplot(x = "FuelType", y = cars_data['Price'], hue = "Automatic", data = cars_data)

#plotting 2 plots on the same window
#split the plotting window into 2 parts
f,(ax_box, ax_hist) = plt.subplots(2, gridspec_kw={"height_ratios" : (0.15,0.85)})
#gridspec : ratio of gridsize height
#split window into 2
#f is a figure
#ax_box : axes for boxplot
#ax_hist : axes for hist

sns.boxplot(cars_data["Price"], ax=ax_box) #boxplot
sns.distplot(cars_data['Price'], ax = ax_hist, kde = False) #hist

