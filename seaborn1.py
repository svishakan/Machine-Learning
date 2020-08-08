# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 20:58:04 2019

@author: Vishakan
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


os.chdir('Datasets')


cars_data = pd.read_csv('Toyota.csv',na_values=['??','????'], index_col = 0)
cars_data.dropna(axis = 0, inplace = True)

#seaborn is a data visualisation library based on matplotlib
#provides a high level interface for attractive and informative graphs
#need to import plt as well, as the seaborn is built on top of plt



#scatterplot of price vs age (lmplot and regplot)
sns.set(style = "darkgrid")
#theme, dark and with grid
#default theme
sns.regplot(x = cars_data['Age'], y = cars_data['Price'], marker = '*')
#regression plot with x and y coordinates
#regression line is fitted into the scatter
#fit_reg = True by default. can be disabled
#relates y and x using a function
#marker changed using marker arg


#scatter plot of PRICE V AGE by FUELTYPE
#using hue parameter, incld. another variable to show 
#the fuel types categories with diff. colors

sns.lmplot(x = 'Age', y = 'Price', data = cars_data, fit_reg=True, hue = 'FuelType', legend = True, palette="Set1")
#phasor grid + regplot is lmplot
#one more variable added using hue
#points are difftd. using color.
#therefore we have a legend mapping and a color palette
#transparency, shape and size can also be used to differentiate the markers



