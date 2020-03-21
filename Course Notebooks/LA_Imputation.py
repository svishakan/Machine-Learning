# -*- coding: utf-8 -*-
"""
@author: GITAA
"""

######################## MODULE: DATA IMPUTATION ##############################

#%%
# Importing pandas for reading data
import pandas as pd

# Importing numpy for numerical operations
import numpy as np

#%%
# Importing data
data=pd.read_csv('GTPvar.csv',index_col=0)


#%%
# Check the missing values columnwise
data.isnull().sum(axis=1)

#%%

# Creates a new variable NApresent which will 
# contain the number of NAs in each row
data['NApresent']=data.isnull().sum(axis=1)

#%%
# No.of variables under each category
data['NApresent'].value_counts()

#%%
# Extract those rows with no NAs and find if the variables 
# have any relationship 
df=data[data.NApresent==0]

# Dropping last column
df=df.drop('NApresent',axis=1)

#%%
# Converting the data to a numpy array
df_mat=df.to_numpy()

#%%

######################## RANK OF MATRIX ###########################

np.linalg.matrix_rank(df_mat)

# SVD decomposition


v,s,u=np.linalg.svd(df_mat.T)

#%%
# Setting tolerance
tol=1e-8

# Removing columns that are lesser than the tolerance
rank=min(df_mat.shape)-np.abs(s)[::-1].searchsorted(tol)

# Choosing the null space relation
A=v[:,rank:]
A=A.T
print(A)

#%%

# Now our task is to run through each record and perform the following:
#   > check how many NAs exist. We can only work with those cases where this number is 3
#   > consider only as many equations (rows of A) as the number of NAs
#   > check which fields have NAs. Those fields would constitute our local 'a' matrix
#   > the rest of the fields having values constitute the constant term when multiplied by the corresponding columns in A
#   > this value corresponds to "-b" in the equation ax = b

#aId collects all the column Ids where NA is present- for each row
#bId collects all the column Ids where NA is not present- for each row



len(data)
len(A)  
  
for i in range(0,len(data)):
    if((data.iloc[i,5]==0) | (data.iloc[i,5]>len(A))):
        continue
    else:
        eqnsneeded=data.iloc[i,5]
        aID=np.empty(0,dtype='int64')
        bID=np.empty(0,dtype='int64')
        for j in range(len(data.columns)-1):
            if(pd.isnull(data.iloc[i,j])):
                aID=np.append(aID,j)
            else: bID=np.append(bID,j)
        a=A[0:eqnsneeded,aID]
        a=np.array(a)
        x1=((data.iloc[i,bID].to_numpy()))
        b2=-A[0:eqnsneeded,bID]
        b=np.dot(b2,x1)
        x=np.linalg.solve(a,b)
        data.iloc[i,aID]=x
        