# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:20:21 2019

@author: GITAA
"""

#=============================================================================
# READING DATA
#=============================================================================

# Importing necessary libraries
import pandas as pd

'''
=============================================================================
 Reading .csv format data
=============================================================================
'''
data_csv = pd.read_csv('Iris_data_sample.csv')

# =============================================================================
# Setting the 1st coulmn 'Unnamed: 0' as index column while reading data
# =============================================================================
data_csv = pd.read_csv('Iris_data_sample.csv',index_col=0)

# =============================================================================
# Replacing ‘??’ and ‘# # #’ as 'nan' values
# =============================================================================

data_csv = pd.read_csv('Iris_data_sample.csv', 
                       index_col=0,na_values=["??"])

data_csv = pd.read_csv('Iris_data_sample.csv',
                       index_col=0,na_values=["??","###"])

'''
=============================================================================
Reading .xlsx format data
=============================================================================
'''
data_xlsx = pd.read_excel('Iris_data_sample.xlsx',
                        sheet_name='Iris_data')

data_xlsx = pd.read_excel('Iris_data_sample.xlsx',index_col=0)

data_xlsx = pd.read_excel('Iris_data_sample.xlsx',
                        index_col=0,
                        na_values=["??","###"])
'''
=============================================================================
Reading .txt format data
=============================================================================

 - Delimitor can be a space or a tab               
 - Try out to see what works  
'''
 
data_txt1 = pd.read_table('Iris_data_sample.txt',delimiter="\t")
data_txt1 = pd.read_table('Iris_data_sample.txt',delimiter=",")
data_txt1 = pd.read_table('Iris_data_sample.txt',delimiter=" ") # correct

#Instead of using read_table(), read_csv() can also be used to read .txt files
data_txt2 = pd.read_csv('Iris_data_sample.txt',delimiter=" ")

# =============================================================================
#   END OF SCRIPT
# =============================================================================
