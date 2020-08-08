# -*- coding: utf-8 -*-
"""
@author: GITAA
"""
'''
    Basic plots using matplotlib library:
    - Scatter plot
    - Histogram
    - Bar plot
    Basic plots using seaborn library:
    - Scatter plot
    - Histogram
    - Bar plot
    - Box and whiskers plot
    - Pairwise plots
'''
# =============================================================================
# Importing necessary libraries
# =============================================================================
import os               # ‘os’ library to change the working directory
import pandas as pd     # ‘pandas’ library to work with dataframes
import numpy as np      # ‘numpy’ library to perform numeric operations
import matplotlib.pyplot as plt # to visualize the data
import seaborn as sns   # to visualize the data
# =============================================================================
# Importing data (replacing special chars with nan values)
# =============================================================================

cars_data = pd.read_csv('Toyota.csv', index_col=0, na_values=["??","????"])

# Removing missing values from the dataframe
cars_data.dropna(axis = 0, inplace=True)

# =============================================================================
# SCATTER PLOT - MATPLOTLIB
# =============================================================================

plt.scatter(cars_data['Age'], cars_data['Price'], c  ='red', )
plt.title('Scatter PLot')
plt.xlabel('Age (months)')
plt.ylabel('Price (Euros)')
plt.show() 

#The price of the car decreases as age of the car increases

# =============================================================================
#  HISTOGRAM - MATPLOTLIB
# =============================================================================
# Histogram with default arguments
plt.hist(cars_data['KM']) 

plt.hist(cars_data['KM'], color = 'red', edgecolor = 'white', bins =5)
"""
Frequency distribution of kilometre of the cars shows that most of the cars have
travelled between 50000 – 100000 km and there are only few cars with more distance travelled
"""
# for any bin value, the minor tick lables remains the same
# only the no.of bars changes

# histogram for the given range of values
plt.hist(cars_data['KM'], color='blue', edgecolor='white', bins=10, range=(5000,15000))
plt.show()

# =============================================================================
# BAR PLOT - MATPLOTLIB
# =============================================================================

counts   = [979, 120, 12]
fuelType = ('Petrol', 'Diesel', 'CNG')  # Set the labels of the xticks
index    = np.arange(len(fuelType))     # Set the location of the xticks

plt.bar(index, counts, color=['red', 'blue', 'cyan'], edgecolor='darkblue')
plt.title('Frequency plot of FuelType')
plt.xlabel('Fuel Type')
plt.ylabel('Frequency')
plt.xticks(index, fuelType,rotation = 90)
plt.show()

"""
Bar plot of fuel type shows that most of the cars have petrol as fuel type
""" 
# =============================================================================
# SACTTER PLOT - SEABORN
# =============================================================================
sns.set(style="darkgrid")

#1. Scatter plot of Price vs Age with default arguments
sns.regplot(x=cars_data['Price'], y=cars_data['Age'])

# By default, fit_reg = True 
# It estimates and plots a regression model relating the x and y variables
 
# 2. Scatter plot of Price vs Age without the regression fit line
sns.regplot(x=cars_data['Price'], y=cars_data['Age'], fit_reg=False)

# 3. Scatter plot of Price vs Age by customizing the appearance of markers
sns.regplot(x=cars_data['Price'], y=cars_data['Age'], 
            marker="*", fit_reg=False)
sns.plt.show()

sns.regplot(x=cars_data['Price'], y=cars_data['Age'], 
            marker="o", fit_reg=False,
            scatter_kws={"color":"green","alpha":0.3,"s":200} )
sns.plt.show()

# 4. Scatter plot of Price vs Age by FuelType
# Using hue parameter, including another variable to show the fuel types 
# categories with different colors

sns.lmplot(x = 'Age', y = 'Price', data = cars_data, fit_reg = False,
           hue = 'FuelType', legend = True, palette="Set1")
sns.plt.show()

# 4. Differentiating categories using markers
sns.lmplot(x = 'Age', y = 'Price', data = cars_data, fit_reg = False,
           hue = 'FuelType', legend = True, markers=["o", "x", "1"])
sns.plt.show() 

# =============================================================================
# HISTOGRAM - SEABORN
# =============================================================================
# 1.Histogram of Age with default kernel density estimate 
sns.distplot(cars_data['Age'] )
 
# 2. Histogram without kernel density estimate
sns.distplot(cars_data['Age'], hist=True, kde=False)

# 3. Histogram with fixed no. of bins
sns.distplot(cars_data['Age'], bins=5 )


# =============================================================================
# BAR PLOT - SEABORN
# =============================================================================

# Frequency distribution of fuel type of the cars
sns.countplot(x="FuelType", data=cars_data)

# Grouped bar plot of FuelType and Automatic
sns.countplot(x="FuelType", data=cars_data, hue = "Automatic")

sns.countplot(y="FuelType", data=cars_data, hue = "Automatic")

sns.countplot(x="FuelType", data=cars_data, palette="Set2")

# =============================================================================
# Box and whiskers plot
# =============================================================================
# 1. Box plot for a numerical varaible
#    Box and whiskers plot of Price to visually interpret the five-number summary

sns.boxplot(y=cars_data["Price"] )

#2. Box and whiskers plot for numerical vs categorical variable 
#   Price of the cars for various fuel types 

sns.boxplot(x = cars_data['FuelType'], y = cars_data["Price"])

#3. Box plot for multiple numerical varaibles
sns.boxplot(data = cars_data.ix[:,0:4])

#4. Grouped box and whiskers plot of Price vs FuelType and Automatic

sns.boxplot(x="FuelType",  y = cars_data["Price"], 
            hue="Automatic", data=cars_data, palette="Set2")

# =============================================================================
# Box-whiskers plot and Histogram
# =============================================================================

# Let’s plot box-whiskers plot and histogram on the same window
# Split the plotting window into 2 parts 

f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
 
# Now, create two plots
sns.boxplot(cars_data["Price"], ax=ax_box)
sns.distplot(cars_data["Price"], ax=ax_hist, kde = False)
 
# Remove x axis name for the boxplot
ax_box.set(xlabel='')


# =============================================================================
# END OF SCRIPT
# =============================================================================
