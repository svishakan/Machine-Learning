import os
import pandas as pd
import numpy as np

os.chdir('Datasets')
cwd = os.getcwd()
print(cwd)

# csv_file = pd.read_csv('filename', index_col=0, na_values=["??","###"])
# excel_file = pd.read_excel()
# text_file = pd.read_table('filename',sep='\t') or pd.read_csv()

#dataframe is a 2D mutable, a tabular data structure with labelled axes

cars_data = pd.read_csv('Toyota.csv',index_col=0)


#two types of copies: shallow copy and deep copy
#samp = cars_data.copy(deep = False) or samp = cars_data
#creates a new variable that shares the reference of the original object. Doesn't create new object. Original will also be changed if the shallow copy changes

#samp = cars_data.copy(deep = True) or samp = cars_data.copy() (default is True for deep)
#creates a new object and isn't linked to the original dataframe.


cars_data1 = cars_data.copy(deep = True)
print(cars_data1.index) #indices of the dataframe
print(cars_data1.columns) #columns of the dataframe
print(cars_data1.size) #total size of dataframe
print(cars_data1.shape) #dimensions of dataframe
print(cars_data1.memory_usage()) #memory used by each column
print(cars_data1.ndim) #no . of axes/array dimensions


#indexing and selecting data (done by dot and slicing operator)
print(cars_data1.head(5)) #returns the first 5 rows. if no val, it returns 5 values(default)
print(cars_data1.tail(3)) #similar to head. returns last n rows of data

#at provides label-based scalar lookups
print(cars_data1.at[4,'FuelType']) #row label and column label specified

#iat provides integer based lookups
print(cars_data1.iat[5,6]) #row,column number specified

#loc[] for accessing group of rows and columns
print(cars_data1.loc[0:5,'FuelType']) #row label and column label
# : represents all rows (usual slicing operator)


#data types: numeric and character types
#numeric : integer, float
#character : category, object (strings)
#category : takes fixed, limited no. of possible values, saves memory. Is a a string
#object : mixed types(num + string). If a column has nan values, then that will default to object datatype. Length or values aren't fixed.
#int64 in pandas = int in python
#float64 in pandas = float in python
# 64 = 64 bits = 8 bytes = mem. allocated to each cell

print(cars_data1.dtypes) #checking datatypes of each column
#print(cars_data1.get_dtype_counts()) #returns counts of unique data types in the DataFrame
print(cars_data1.dtypes.value_counts()) #same as above. Since above will be deprecated, use this.
print(cars_data1.select_dtypes(exclude=[object])) #returns a subset of the columns from dataframe based on the column datatypes

#info() returns concise summary of dataframe
print(cars_data1.info())
#if any error in the way the datatype is assigned, we can look it up here and change it.
#here KM is object, and not integer. (For example) (KM HP MetColor Automatic Doors are wrongly interpreted)

print(np.unique(cars_data1['KM'])) #gives the unique values of the array
#On checking, we have the '??' as a data in KM. Which is why KM has been read as object instead of int64
#similarly in HP we have '????'
#in MetColor, we have 0. and 1. which is why it has been read as float64
#in Doors we have 'five' 'four' 'three' values instead of 5 4 and 3. That is why it is being referred to as an object.

#Now, we need to change these appropriately so that it can be changed back to its appropriate datatype.

cars_data = pd.read_csv('Toyota.csv',index_col=0,na_values=['??','????'])
print(cars_data.info())
#now KM and HP became float64 instead of object since we replaced the ?? and ???? with nan
#therefore there are now some missing values. the no. of entries reduce if you observe.

#converting the datatype explicitly
#using astype()

cars_data1 = cars_data.copy()

cars_data['MetColor'] = cars_data['MetColor'].astype('object')
cars_data['Automatic'] = cars_data['Automatic'].astype('object')
print(cars_data.info())
#converted to object & object from float64 and int64 resly.


#nbytes() is used to get the total bytes consumed by the elements of the column
#ndarray.nbytes

#if object data type
print(cars_data['FuelType'].nbytes)
#gives 11488 bytes

#if category data type,
#then, output gives 1436 bytes! Almost 10 times reduction.
print(cars_data['FuelType'].astype('category'))

print(cars_data.info())

#cleaning the data of Doors column
#i.e remove five four three data
#using replace() command

#try out numpy.where()
#np.where() returns the indices of elements that satisfy a given condition in an input array
#can be used here

doors = np.array(cars_data['Doors'])
print(doors)
three_doors = np.where(doors=="three")
print(three_doors)
print(doors[three_doors]) #so here we got the values corresponding to doors = "three"
#this can be extended to the whole DataFrame and the whole row of information can be obtained as
#dataframe objects can use numpy methods.

cars_data['Doors'].replace('three',3,inplace=True)
cars_data['Doors'].replace('four',4,inplace=True)
cars_data['Doors'].replace('five',5,inplace=True)
#inplace is used to reflect the changes to the working dataframe.

#after doing this it'll become objects types
print(cars_data.info())
#now we have converted the strings to numeric type
#now convert the object type of Doors to int64

cars_data['Doors'] = cars_data['Doors'].astype('int64')
print(cars_data.info())

#to check count of missing values present in the columns
print(cars_data.isnull().sum())
#isnull() returns True or False
#can also use the isna() function
#sum() sums up the no. of True's returned

#now we have to fill up those missing values


cars_data2 = cars_data.copy()
cars_data3 = cars_data2.copy()

#tuples with missing values
missing = cars_data2[cars_data2.isnull().any(axis=1)]
#1 represents column in axis
print(missing)


#ways to fill in the missing values:
# fill by mean/median for numerical variable
#fill in by maximum count(mode) for categorical variable
description = cars_data.describe() #use the variable explorer to view
print(description)
#generates descriptive statistics that summarize the 
#central tendency, dispersion and shape of a dataset's distribution
#excluding NaN values.


#looking at the description, we can use the 
#mean for age since median(50%) and mean are similar
#for KM, use Median as mean and median are very different
age_mean = cars_data2['Age'].mean()
km_median = cars_data2['KM'].median()
#to fill in the NaN values
cars_data2['Age'].fillna(age_mean,inplace = True)
cars_data2['KM'].fillna(km_median,inplace = True)
#similarly, we use mean for HP
#inplace (changes in the existing dataframe)
cars_data2['HP'].fillna(cars_data2['HP'].mean(),inplace = True)

print(cars_data2.isnull().sum())
#now only categorical variables metcolor and fueltype has the empty values
#use MODE for categorical variables     
#value_counts() returns counts of unique elements in desc. order frequency

print(cars_data2['FuelType'].value_counts())
#index[0] has the most occurring element. Here, it is Petrol

cars_data2['FuelType'].fillna(cars_data2['FuelType'].value_counts().index[0], inplace = True)
print(cars_data2['MetColor'].mode()) #prints index with the mode value (more than 1 if bimodal/more than one node)
cars_data2['MetColor'].fillna(cars_data2['MetColor'].mode()[0], inplace = True)
print(cars_data2.isnull().sum())
#now there are no empty values.

#to do the whole thing in one stretch using a lambda function
#using apply()
#applies operations row wise/column wise

cars_data3 = cars_data3.apply(lambda x:x.fillna(x.mean()) if x.dtype == 'float' or x.dtype == 'int' else x.fillna(x.value_counts().index[0]))
#numerical variable replaced with mean
#categorical variable replaced with mode
print(cars_data3.isnull().sum())


#tinkering with if-else conditions and loops
cars_data1.insert(10,"Price_Class","") #creating new column price class with empty values
for i in range(0,len(cars_data1['Price'])):     #to iterate over the table contents
    if(cars_data1['Price'][i]<=8450):
        cars_data1['Price_Class'][i]="Low"
    elif(cars_data1['Price'][i]>11950):
        cars_data1['Price_Class'][i]="High"
    else:
        cars_data1['Price_Class'][i]="Medium"


#similarly, one can use a while loop too
print(cars_data1['Price_Class'].value_counts())


#functions in Python
#function to convert the Age of car from months to years and put it in another column
cars_data1.insert(11,"Age_Years",0)

def month_to_year(val):
    conv = val/12
    conv = round(conv,1) #rounding off to 1 decimal point
    return conv

cars_data1["Age_Years"]=month_to_year(cars_data1['Age'])


#converting kilometres to km_per_month
cars_data1.insert(12,"KMPM",0)


def KMPM_MTY(val1,val2): #KMPM and month to year
    conv = val1/12
    print(val2)
    conv = round(conv,1)
    ratio = val2/val1
    ratio = round(ratio,2)
    return [conv,ratio]

cars_data1["Age_Years"],cars_data1["KMPM"] = KMPM_MTY(cars_data1['Age'],cars_data1['KM'])



#exploratory data analysis

cars_data4 = cars_data.copy()

#frequency table: 
#crosstab() a simple cross-tabulation of one or more factors
#by default computes a freq. table of the factors
#we are looking at the freq. table for the FuelType data
print(pd.crosstab(index = cars_data4['FuelType'],columns = 'count',dropna = True))

#two-way tables using crosstab()
print(pd.crosstab(index = cars_data4['Automatic'],columns = cars_data4['FuelType'],dropna = True))
#values only for which both Automatic and FuelType are not NA are considered.
#1 Automatic, 0 Manual


#two-way table, joint probability using crosstab()
#joint prob. is the likelihood of 2 indep. events happening at the same time
print(pd.crosstab(index = cars_data4['Automatic'],columns = cars_data4['FuelType'],normalize = True, dropna = True))
#normalize : table values converted from numbers to proportions

#two-way table, marginal probability using crosstab()
#marginal prob. is the prob. of occurrence of the single event
print(pd.crosstab(index = cars_data4['Automatic'],columns = cars_data4['FuelType'], margins = True, dropna = True, normalize = True))

#prints individual indep. proportions along with the row and column sums(proportions)
#Example : 0.94 of row1 is that the prob. of car being CNG/Diesel/Petrol when gearbox is of Manual type.

#conditional prob. using crosstab()
#prob. of event A given B has occurred.
#given type of gearbox, prob. of diff. fuel type
print(pd.crosstab(index = cars_data4['Automatic'], columns = cars_data4['FuelType'], dropna = True, margins = True, normalize = 'index'))
#normalized based on index value
# Given Manual Gearbox, Prob. of getting CNG Fuel Type is 0.011 (inference example from the crosstab)

#normalize based of column value
print(pd.crosstab(index = cars_data4['Automatic'], columns = cars_data4['FuelType'], dropna = True, margins = True, normalize = 'columns'))
# Given Petrol FuelType, car has Manual gearbox is 0.9397 (Example interpretation from the crosstab)


#Correlation
#checking strength of association between 2 variables
#visual representation: scatterplot
# a value between -1 and 1. val ~ 1 is a high correlation
# 0 : no correlation
# -1 : negative correlation

#pairwise correlation of columns excluding NA values
#excluding categorical variables to find Pearson's correlation
#to check the strength of association between 2 numerical variables
#kindall-rankall, spearman for ordinal variables

#getting numerical DATAFRAME from the CARS DATAFRAME
numerical_data = cars_data2.select_dtypes(exclude=[object])
#by default we used Pearson's correlation method here.
corr_matrix = numerical_data.corr()
#other correlation methods can be given by using self,method = 'Pearson' arguments for the corr function
#principal diagonal values are 1 because Price,Price, Age,Age etc. are related the same to themselves
#there's a strong correlation between Age and Price



