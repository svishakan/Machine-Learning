# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 19:04:45 2019

@author: Vishakan
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.chdir('Datasets')

cars_data = pd.read_csv('Toyota.csv',na_values=['??','????'], index_col = 0)

#scatterplot plt.scatter()
#set of points that represent values for 2 diff. variables plotted on a 2d graph
#is also called as a correlation plot

cars_data.dropna(axis = 0, inplace = True)
#removing NA value records from the same dataframe and making it permanent

plt.scatter(cars_data['Age'], cars_data['Price'], c = 'red')
#x coordinate, y coordinate, color 
plt.title('Scatterplot of Price v. Age of Cars')
plt.xlabel('Age in Months')
plt.ylabel('Price in Euros')
plt.show()
#relationship suggests: price of car decreases as age increases.
#points are markers
#vertical lines are ticks
#values are called labels (0,10,20) etc.

#histogram plt.hist()
#graphical representation of data using bars of diff. heights
#freq. distribution for NUMERICAL DATA
#groups numbers into ranges(bins), and output of each bin is given

plt.hist(cars_data['KM']) #default args
plt.title('Histogram of KM')
plt.show()
#better histogram
plt.hist(cars_data['KM'], color = 'green', edgecolor = 'white', bins = 5)
#making bins distinguishable and also limiting the no. of bins
plt.title("Histogram of KM")
plt.xlabel('Kilometres')
plt.xlabel('Frequency')
plt.show()



#bar plot plt.bar()
#represents categorical data as rectangular bars with length proportional to counts they represent
#freq. distribution of CATEGORICAL variables
#compare sets of data between diff. groups

counts = [979,120,12] #counts of each categories. can be obtained from info()
fuelType = ('Petrol','Diesel','CNG') #above can also be substituted with value_counts
index = np.arange(len(fuelType)) #[0,1,2]

#index : x coordinate
#counts: height of coordinate
print(cars_data['FuelType'].value_counts()) #gives the values for the counts list above
plt.bar(index, counts, color =['red','blue','cyan'])
plt.title("Bar Plot of Fuel Types")
plt.xlabel('Fuel Types')
plt.ylabel("Frequency")
plt.xticks(index, fuelType, rotation =90) #setting labels for the bar plot as the x-coordinate are number values, which don't make sense
#rotation: labels are rotated by 90 degrees
plt.show()






















