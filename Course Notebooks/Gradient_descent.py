# -*- coding: utf-8 -*-
"""
@author: GITAA
"""

######################## MODULE: BINARY CLASSIFICATION ##############################


#%%

#Importing the required library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
#Reading the datasets
train_data = pd.read_csv('engineTest.csv')
test_data = pd.read_csv('engineTestCheck.csv')

#%%
#Printing the columns 
print(train_data.columns)
print(test_data.columns)

#%%
train_data['Result'].value_counts()


#%%
"""
In this example the first two variables X1 and X2 are used for the 
optimization in order to demonstrate the result graphically
"""

# Creating a new data frame then adding a column for the intercept 
#'X0' and subseting 'X1','X2' and 'Result' from the train_data

train_data1 = pd.DataFrame({'X0': np.repeat(1, len(train_data))})

train_data1[['X1','X2', 'Result']] = train_data[['X1','X2', 'Result']]


#%%

# n_k is the variable created, will finally be equation of the line
n_k = np.repeat(10,3)


# n_kprev is created to enable updates to the n_k value in the loop
n_kprev = np.repeat(0,3)

# step length
cLearn = 0.1 

# variable is created to verify if all samples meet condition
updateCounter = True

# classification criteria
maxIteration = 2000
iteration = 1

#%%

# grouping the train_data according to the 'Result'
grouped_data = train_data1.groupby(by='Result')

for key in grouped_data.groups.keys():
    data = grouped_data.get_group(key)
    x =data ['X1']
    y =data ['X2']
    plt.scatter(x,y,label = key)
    plt.xlim(0,80)
    plt.ylim(-5,80)
    plt.legend()
    
    slope = -1*n_k[1]/n_k[2]
    intercept = -1*n_k[0]/n_k[2]
    X = np.arange(80)
    Y=slope*X+intercept
    plt.plot(X,Y,c ='red')
plt.show()


#%%


while updateCounter and iteration < maxIteration:
    updateCounter = False
    for i in range(len(train_data1)):
        
        # assignment of initial guess to the previous value for gradient search algorithm
        n_kprev = n_k
        
        pred = np.array(train_data1.iloc[i,0:3]).dot(n_k)
        #print(pred)
        if(train_data1.Result[i] == "Pass" and pred < 0):
            #This condition checks if the "passed" samples are classified 
            #properly
            n_k = n_kprev + cLearn *((np.array(train_data1.iloc[i,0:3] )).T)
            #updated counter changed to true to reflect the fact that 
            #equation was modified
            updateCounter = True
        elif(train_data1.Result[i] == "Fail" and pred > 0):
                
            #This condition checks if the "failed" samples are classified 
            #properly
            n_k = n_kprev - cLearn *((np.array(train_data1.iloc[i,0:3] )).T)
            #updated counter changed to true to reflect the fact that 
            #equation was modified
            updateCounter = True 
            
        if(i % 15 == 0 and iteration <= 2):
            grouped_data = train_data1.groupby(by='Result')
            for key in grouped_data.groups.keys():
                data = grouped_data.get_group(key)
                x =data ['X1']
                y =data ['X2']
                plt.scatter(x,y,label = key)
                plt.xlim(0,80)
                plt.ylim(-5,80)
                plt.legend()
                
                slope = -1*n_k[1]/n_k[2]
                intercept = -1*n_k[0]/n_k[2]
                X = np.arange(80)
                Y=slope*X+intercept
                plt.plot(X,Y,c ='red')
            plt.show()
    #print(iteration)
    iteration = iteration +1

#%%



test_data.columns

test_data1 = pd.DataFrame({'X0': np.repeat(1, len(test_data))})

test_data1[['Engine','X1','X2']] = test_data[['Unnamed: 0','X1','X2']]

test_data1.sort_index(axis =1, inplace =True)

test_result = np.array(test_data1.iloc[:,1:4]).dot(n_k)
test_data1["Test_Result"] = test_result

test_data1["Predicted_Result"] = np.where(test_data1['Test_Result'] > 0 ,"Pass","Fail")

grouped_data = test_data1.groupby(by='Predicted_Result')
for key in grouped_data.groups.keys():
    data = grouped_data.get_group(key)
    x =data ['X1']
    y =data ['X2']
    plt.scatter(x,y,label = key)
    plt.xlim(0,80)
    plt.ylim(-5,80)
    plt.legend()
    
    slope = -1*n_k[1]/n_k[2]
    intercept = -1*n_k[0]/n_k[2]
    X = np.arange(80)
    Y=slope*X+intercept
    plt.plot(X,Y,c ='red')
plt.show()


