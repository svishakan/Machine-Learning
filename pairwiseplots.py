# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:33:49 2019

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


#pairwise plots
#used to plot pairwise relationships in a dataset
#creates scatterplots for joint relationships and histograms for univariate distributions
#all relationships for all variables, colored based on Fueltype
#diagonals are all histograms because they are univariate relationships
sns.pairplot(cars_data, kind = "scatter", hue = "FuelType", diag_kind = "hist")
#diag_kind unspecified leads to LinAlgError
plt.show()