#################  Intro to Numpy  ############################
###															###
###															###
###   				Introduction to Numpy					###
###															###
###															###
###############################################################

import numpy as np

#################  lets create an array from a list  ###########################
my_list1 = [1,2,3,4]
my_list2 = [11,22,33,44]
my_array = np.array(my_lists)

my_array.shape # will return (2L, 4L) because its a 2x4 array

my_array2.dtype # will return dtype('int32') since its an integer array

#################  short cut to common array types  ###########################

np.zeros([5,5]) # creates a 5x5 zero array
np.eye(5) # creates a 5x5 identity matrix
np.ones([5,5]) # creates a 5x5 matrix with all elements equal to zero
np.arange(2,30,3) # creates a single row array in a range from 2 to 30 incremented by 3
np.arange(2,30,3).reshape([5,2]) # creates a 5x2 array with elements in a range from 2 to 30 incremented by 3


#################  some array calculations  ###########################

from __future__ import division # use this when working with Python 2.x version. 

arr1 = np.array([[1,2,3,4],[8,9,10,11]])

arr1 * arr1 # returns element wise multiplication
arr1 - arr2 # element wise subtraction
1/arr1 # reciprocal of all elements in an array
arr[0][1]*arr[4][1] #multiplying individual elements of an array
np.dot(u,v) # performs dot product (scalar output) of 2 numpy arrays u,v
np.dot(U,V) # performs matrix multiplication of 2 matrices U & V

arr = np.arange(50).reshape([10,5]) #returns a 10x5 array
arr.T # transpose 
np.dot(arr,arr.T) # matrix multiplication

from numpy.linalg import inv
AAtInv = np.linalg.inv(AAt) # returns inverse of matrix AAt

anarr = np.arange(12).reshape(4,3) 
'''
anarr will be
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]])
'''
anarr.swapaxes(1,0) # swap axis .. swaps axis
'''
swapping axis will return
array([[ 0,  3,  6,  9],
       [ 1,  4,  7, 10],
       [ 2,  5,  8, 11]])
'''	   

np.sqrt(arr) #square root of elements in an array
np.exp(arr) # raises every element in the array to the e-power

np.add(A,B) # element wise addition of 2 arrays
np.maximum(A,B) #maximum values between 2 arrays at each index

website = 'http://docs.scipy.org/doc/numpy/reference/ufuncs.html#available-ufuncs' # go here for more details on available functions for arrays.
import webbrowser
webbrowser.open(website)

#################  array indexes ###########################

arr = np.arange(1,11)

arr[8:11] = [50,60,70] # replacing specific indexes of an array with new values
slice_of_arr = arr[0:6] # arr[0:6] can now be referred to by slice_of_array. slice_of_array is NOT a new array. Its simply a pointer to certain elements of array 'arr'
slice_of_arr[:] = 99 #replace all values of 'arr' array referred to as slice_of_array by a value = 99
arr_2d_slice = arr_2d[:2,1:] # same example as above, but with a 2d array. Index being referred to here are all rows until row 3, and all columns starting from 2nd column

arr_copy = arr.copy() #to make an actual copy of an array, use .copy() method

arr2d = np.zeros([10,10]) #lets create a 10x10 2d array
ln = arr2d.shape[0] #ln will equate to 10, since we are asking for shape of the first row of arr2d

for i in range(ln):
    arr2d[i]=i #now lets replace every element of the 10x10 2d array

fancy_arr = arr2d[[6,4,2,9]] 
'''
fancy_arr will be the following array

array([[ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.],
       [ 4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.,  4.],
       [ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.],
       [ 9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.,  9.]])
'''

#################  hstack and vstack ###########################

zeros = np.zeros(8)
ones = np.ones(8)
print 'zeros:\n{0}'.format(zeros)
print '\nones:\n{0}'.format(ones)

zerosThenOnes = np.hstack((zeros,ones))   # A 1 by 16 array. 
zerosAboveOnes = np.vstack((zeros,ones))  # A 2 by 8 array

print '\nzerosThenOnes:\n{0}'.format(zerosThenOnes)
print '\nzerosAboveOnes:\n{0}'.format(zerosAboveOnes)

#################  some more cool stuff you can do with arrays ###########################

# some simple plots
import matplotlib.pyplot as plt
%matplotlib inline #enables inline plot in an python notebook

points = np.arange(-5,5,0.01)
dx,dy = np.meshgrid(points,points) #meshgrid creates a grid.

z = (np.sin(dx) + np.sin(dy)) #z is a sine function

plt.imshow(z) # imshow() plots the array
plt.colorbar() # displays the color bar
plt.title("Plot of sin(x)+ sin(y)") # adds title to the plot

# using the where feature of numpy. Inline if else feature
A = np.array([1,2,3,4])
B = np.array([100,200,300,400])
Cond = np.array([True,True,False,False])

answer2 = np.where(Cond,A,B) # if Cond is true, elements of array answer2 equal elements of array A, else elements of array answer2 equal elements of array B

# another python way to do the above is as follows
answer = [(A_val if cond else B_val) for A_val, B_val, cond in zip(A,B,Cond)]

# another example of where

from numpy.random import randn # from random library import randn (random normal)
arr = randn(5,5)
np.where(arr<0,0,arr) # read this as 'if condition arr<0 is true, replace with 0, else retain value'

# adding sum of elements
arr.sum() #sums all elements
arr.sum(0) # add the columns
arr.sum(1) # add the rows
arr.mean() # mean of the columns
arr.std() # std deviance
arr.sort() # sorts every row of an array

# boolean arrays
bool_arr = np.array([True,False,True])
bool_arr.any() # will return true if any item in boolean array is true
bool_arr.all() # will return true ONLY if all items in boolean array is true

# non numeric arrays
countries = np.array(['france','germany','usa','russia','usa','mexico','germany'])
np.unique(countries) # unique elements in an array
np.in1d(['france','usa','sweden'],countries) # in1d returns boolean array and a true for all elements in an array


#################  saving and loading arrays ###########################

np.save('myarray',arr)
np.load('myarray.npy')

np.savez('ziparray.npz',x=arr1,y=arr2) # saving multiple arrays
archive_array = np.load('ziparray.npz')
archive_array['x'] # returns array 'x'
archive_array['y'] # returns array 'y'

np.savetxt('mytxtarray.txt',arr,delimiter=',') #save as a txt
arrnew = np.loadtxt('mytxtarray.txt', delimiter = ',')


###############################################################
###															###
###															###
###   				Dense vectors (pyspark)					###
###															###
###															###
###############################################################

'''
PySpark provides a DenseVector class which allows you to more efficiently operate and store these sparse vectors
DenseVector is used to store arrays of values for use in PySpark. DenseVector actually stores values in a NumPy array and delegates calculations to that object. 
You can create a new DenseVector using DenseVector() and passing in a NumPy array or a Python list
'''

from pyspark.mllib.linalg import DenseVector

# Create a numpy array 
numpyVector = np.array([-3, -4, 5])
print '\nnumpyVector:\n{0}'.format(numpyVector)

# Create a DenseVector 
myDenseVector = DenseVector(np.array([3.0, 4.0, 5.0]))
# Calculate the dot product between the two vectors.
# One of the vectors here is a numpy array, and the other a densevector
denseDotProduct = DenseVector.dot(DenseVector(numpyVector),myDenseVector)

print 'myDenseVector:\n{0}'.format(myDenseVector)
print '\ndenseDotProduct:\n{0}'.format(denseDotProduct)

#################  Intro to Panda  ############################
###															###
###															###
###   				Introduction to Panda					###
###															###
###															###
###############################################################


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import randn

#################  Series basics  #############################
###															###
###															###
###   				Series basics							###
###															###
###															###
###############################################################

ww2_cas = Series([8700000,4300000,3000000,2100000,400000],index=['USSR','Germany','China','Japan','USA']) #create a series with custom index using Series method
countries = ['USSR','Germany','China','Japan','USA']
obj2 = Series(ww2_dict,index=countries) # another way to create a series

ww2_cas.values # returns series values
ww2_cas.index # returns series index values
ww2_cas['USA'] #extract series value with given index
ww2_cas[['USA','China']] # returns values corresponding to index in the list
ww2_cas[4] # returns 5th value in the series
ww2_cas[0:3] # returns first 4
ww2_cas[-1]# returns last value in the series

ww2_cas[ww2_cas>4000000] #extract series value with given condition
ww2_cas['USSR'] = 1 # replaces the value corresponding to the index

'USSR' in ww2_cas # returns boolean true or false
ww2_dict = ww2_cas.to_dict() # to.dict() method converts series to dictionary
ww2_series = Series(ww2_dict) #convert back to Series using the series method

pd.isnull(obj2) # returns index that has NaN as value
pd.notnull(obj2) # opposite

ww2_series + obj2 # new series returned by adding values on the basis of index

obj2.name = "world war 2 casualties"  #name your series
obj2.index.name = 'Countries' # name your index

ser1 = Series([1,2,3,4],index=['A','B','C','D'])
ser2 = ser1.reindex(['A','B','C','D','E','F'], fill_value = 0) # reindex - does exactly that. Adds new indexes to a series. Use fill_value, else the values will be NA.

## forward fill and backward fill of indexes in the examples below
ser4 = Series(['USA','Mexico','Canada'],index = [0,5,10]) 
''' output is 
0        USA
5     Mexico
10    Canada
dtype: object
'''

ser4.reindex(range(15),method='ffill') #ffill is forward fill from index 0. range is python's builtin function. arange is numpy's method
'''
0        USA
1        USA
2        USA
3        USA
4        USA
5     Mexico
6     Mexico
7     Mexico
8     Mexico
9     Mexico
10    Canada
11    Canada
12    Canada
13    Canada
14    Canada
dtype: object
'''
ser4.reindex(range(15),method='bfill') #bfill is backward fill from the last index ... which is 10 for Series ser4

'''
0        USA
1     Mexico
2     Mexico
3     Mexico
4     Mexico
5     Mexico
6     Canada
7     Canada
8     Canada
9     Canada
10    Canada
11       NaN
12       NaN
13       NaN
14       NaN
dtype: object
'''

ser1.drop('b') # drops index 'b' and its associated value

ser1.unique() # returns unique values within a series
ser1.value_counts() # returns counts of values in a Series
'''
w    3
y    2
a    1
z    1
x    1
dtype: int64
'''

# hierarchical indexes are illustrated by these examples

ser = Series(randn(6), index = [[1,1,1,2,2,2],['a','b','c','a','b','c']])
'''
1  a    0.187640
   b    0.792968
   c   -0.317989
2  a   -0.178000
   b   -0.243812
   c   -0.451486
dtype: float64
'''
ser.index
'''
MultiIndex(levels=[[1, 2], [u'a', u'b', u'c']],
           labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
'''

ser[1]

'''
a    0.187640
b    0.792968
c   -0.317989
dtype: float64
'''
ser[2]
'''
a   -0.178000
b   -0.243812
c   -0.451486
dtype: float64
'''

ser[:,'a'] # return all from primary index, but use secondary index = 'a'
ser[1,'a'] # returns value at index 1 (primary), 'a' (secondary)

df = ser.unstack() # converts hierarchical index series into dataframe with primary index as rows, and secondary index as columns

#combine_first() method
Series(np.where(pd.isnull(ser1),ser2,ser1), index = ['x','y','z','q','r','s']) #Series meets numpy where meets panda's isnull() method
# the above statement sates where ser1 values are NaN, use ser2 values, else use ser1 values
ser1.combine_first(ser2) #combine_first() does the same

df1.combine_first(df2) # does the same with dataframes. 

ser1.replace(1,10) # replace '1' in your series with '10'
ser1.replace(1,np.nan) # replace '1' in your series with NaN
ser1.replace([1,4],[100,400]) # replace value (1 and 4) with (100 and 400)
ser1.replace({4: 'clown' , 2: 'owl'}) # replace 4 with clown, and 2 with owl

###############################################################
###															###
###															###
###   				DataFrame basics						###
###															###
###															###
###############################################################

# the key method here is DataFrame()

import webbrowser
website = 'https://en.wikipedia.org/wiki/NFL_win%E2%80%93loss_records'
webbrowser.open(website)

# open the browser and copy the data frame on your clipboard

nfl_frame = pd.read_clipboard() # copy clipboard into a dataframe

nfl_frame.columns # column names
nfl_teams = nfl_frame['Team'] # extract data from column 'Team'. This data is now a series.
nfl_teams_list = nfl_teams.tolist() # convert the series into a list

DataFrame(nfl_frame,columns = ['Team', 'First Season','Total Games']) #subset only the required columns

nfl_frame.head(3) # return top 3 rows
nfl_frame.tail(4) # return bottom 4 rows

nfl_frame.ix[3] # returns object for index 3 using .ix

nfl_frame['Stadium'] = "Levi's Stadium" # returns subset of rows where column = "Levi's Stadium"

nfl_frame['Stadium'] = np.arange(1,6) # replace values in a columns

nfl_frame['Stadium'] = Series(["levi's stadium","AT&T"],index = [4,0]) # replace values in a columns at specific indexes. The rest will be NAs

del nfl_frame['Stadium'] # delete a column

datadic = {'City':['SF','LA','NYC'], 'Population':[837,3800,8400]}
city_df = DataFrame(datadic) ## create dataframe from dictionary

# dropping/deleting rows and columns
df1.drop(['col1','col2'],axis =1) # another way to delete columns. But here you can delete multiple columns. To delete columns use axis = 1.

df1.drop(['A']) # drops row with index 'A'

# more ideas on sub setting a dataframe

df_n = df2[['q','w']] # subset df2 with the columns defined in the index and return to new df called df_n

df2[df2['t']>8] # return only subset of dataframe df2 where column 't' > 8
new_df = df2[df2.factor_column != 'A_factor'] # subset data where a column does not contain 'A_factor'

df2[df2>10] # will return the entire dataframe df2, but will have NaN in places where df2 values are less than 10

df2.ix[[1,2,3,4],[1,4]] # will return a subset of rows 2 - 5 and columns (2 and 5)

###############################################################
###															###
###															###
###			   			IMPORTING DATA						###
###															###
###															###
###############################################################

'''
read csv
'''

dframe = pd.read_csv('file_name.csv', header = None)
dframe = pd.read_csv('file_name.csv', header = None, nrows = 20)
dframe = pd.read_table('file_name.txt', sep = ';' , header = None)

dframe = pd.to_csv('output_file_name.csv')

'''
read html
pip install beautiful-soup
pip install html5lib
'''

url = 'https://www.fdic.gov/bank/individual/failed/banklist.html'
from pandas import read_html

dframe_list = pd.io.html.read_html(url) # read from url and puts the data into a list of dataframe objects
dframe = dframe_list[0]

'''
read excel
pip install xlrd
pip install openpyxl
'''
xlsfile = pd.ExcelFile('input_file.xlsx')
dframe = xlsfile.parse('excel_sheet_name')

'''
read clipboard
'''

dframe = pd.read_clipboard() # copy clipboard into a dataframe

###############################################################
###															###
###															###
###			   		DATAFRAME OPERATIONS					###
###															###
###															###
###############################################################

# go to http://pandas.pydata.org/pandas-docs/stable/cookbook.html for several examples

df3 + df4 #adds dataframes
df4.add(df3,fill_value=0) # does the same thing, and replaces NaNs with 0

ser3 = df3.ix[0] # forming a series from a dataframe. Here the first row is returned as axis

ser3.sort_index() # sorts according to index

ser5 = ser4.order() # sorts according to value, but is NOT in place

ser4.sort() ## in place sorting 

df1.sum() #sum columns

df1.sum(axis = 1) # sum rows

df1.min() # minimum values across columns

df1.idxmin() #index of the minimum values

df1.cumsum() # returns dataframe with cumulative sums across columns

df1.describe() # returns summary stats across columns

df.drop_duplicates() # drops duplicate rows

df.drop_duplicates(['colname']) # keeps only one unique row of 'colname' irrespective of non-duplicates in other columns. Retains the first value uncovered

df.drop_duplicates(['colname'],take_last = True) # retains last value

###############################################################
###															###
###															###
###			   			DATAFRAME NANS						###
###															###
###															###
###############################################################

# lets creare a dataframe
na = np.nan 
df2 = DataFrame([[1,2,3,na],[2,na,5,6],[na,7,na,9],[1,na,na,na]], columns = ['A','B','C','D'])

# now manage NaNs
df2.dropna(thresh = 2) # drop rows with more than '2' missing data points
df2.fillna(1) # fill NaNs with 1
df2.fillna({'A':0,'B':10,'C':20,'D':30}) # custom ways of filling NaNs across columns
df2.fillna({'A':0,'B':10,'C':20,'D':30},inplace = True) # replaces df2 inplace 

###############################################################
###															###
###															###
###			   			DATAFRAME INDEXING					###
###															###
###															###
###############################################################

#reindex examples below
df = DataFrame(randn(25).reshape([5,5]),index=['A','B','D','E','F'], columns = ['col1','col2','col3','col4','col5'])
df2 = df.reindex(['A','B','C','D','E','F'], fill_value = 0) # reindex method for DataFrame
new_columns = ['col1','col2','col3','col4','col5','col6']
df2.reindex(columns = new_columns, fill_value=0) # reindex with new column names

# hierarchical indexing on dataframes
# in the example below both columns and indexes have multiple hierarchies
df2 = DataFrame(np.arange(16).reshape(4,4),index=[['a','a','b','b'],[1,2,1,2]],columns=[['NY','NY','SF','SF'],['cold','hot','cold','hot']])
# a more elegant way of creating hierarchical indexes using MultiIndex
hier_cols = pd.MultiIndex.from_arrays([['Ny','Ny','La','La'],[1,2,1,2]], names = ['city','some_values'])
df2 = DataFrame(np.arange(16).reshape(4,4), columns = hier_cols)
df2

df2.columns.values # output column names
df2.index.names = ['INDEX_1','INDEX_2'] # name or rename an index
df2.columns.names = ['Cities','Temp'] # name or rename column indexes 

# other ways to rename indexes or columns
df.rename(index = {'Old_Name': 'New_Name'},
         columns = {'Old_Col_Name': 'New_Col_Name'},
         inplace = True)

df2.swaplevel('Cities','Temp', axis = 1) # swamp index relationships. (axis =1 is for columns)
df2.swaplevel('Cities','Temp', axis = 1) # swamp index relationships. (axis =1 is for columns)

df2.sortlevel(0) #sorts according to index. df2.sortlevel(1) will sort according to secondary index

df2.sum(level = 'Temp', axis = 1) # sum across 'columnar' index = "Temp" (columnar because of axis = 1)

df.reset_index(level=0, inplace = True) # converts index into a column. 
# This is useful if index has relevant information like date. In which case reset_index(level = 0) converts that date into a column

# Using loc (location) function
df['a_column'].loc[df['a_column'] > 0] = 'I am not zero' # loc(ate) rows where df['a_column'] > 0, and set the value of those rows to 'I am not zero'
df['a_column'].loc[df['a_column'] == 0] = 'I am zero'

###############################################################
###															###
###															###
###			   			DATAFRAME MERGING					###
###															###
###															###
###############################################################

pd.merge(df1,df2) # merges data across common keys. Inner join

pd.merge(df1,df2,on='key') # as suggested merges on column name that you specify. Inner join

pd.merge(df1,df2,on='key',how='left') #left outer join. Returns all rows from the left (df1), and matching elements from the right

pd.merge(df1,df2,on='key',how='right') #right outer join

pd.merge(df1,df2,on='key',how='outer') # full outer join

pd.merge(df_left,df_right,on=['Key1','Key2'],how = 'outer') #merging on 2 keys/columns

pd.merge(df_left,df_right,on='Key1',suffixes = ('_lefty','_righty')) # your choice of suffix on common Key2

pd.merge(df_left,df_right,left_on='key',right_index=True) # merging using column name and index

pd.merge(df_left_hr, df_right_hr, left_on=['key1','key2'],right_index=True) # merging using 2 columns and index

pd.merge(df_left_hr, df_right_hr, left_on=['key1','key2'],right_index=True, how='left') # same as above but using left outer join

pd.concat([df3,df4], axis = 0) # concatenate two data frames on rows (axis = 0)

pd.concat([df3,df4], axis = 0,ignore_index=True) # creates new incremental index (ignores orginal dataframe index)

# merging using map method
dframe_1['new_column'] = dframe_1['common_column'].map(series_1) #add new_column to dframe1 by merging the index of series_1 with common_column at dframe_1
dframe_1['new_column'] = dframe_1['common_column'].map(dictionary_1) # here you merge with key on the dictionary
#Example
dframe['SingleStorey'] = dframe['Stories'].map({'1':'yes','2':'no'})

###############################################################
###															###
###															###
###			   				PIVOTING						###
###															###
###															###
###############################################################

## stacking and unstacking a data frame

df1 = DataFrame(np.arange(8).reshape(2,4),
               index = pd.Index(['LA','SF'],name='city'),
               columns= pd.Index(['A','B','C','D'], name = 'letters')) # pd.Index enables naming of the columns or index

df_st = df1.stack() # pivots rows into columns

df_st.unstack() # unpivots the above operation

df_st.unstack('city') # will ensure that 'city' are the columns

# to go from long data frame to wide data frame, we can use the pivot function
# the pivot function is also useful in the excel pivot kind of way
dframe.pivot_table(index=['zone'], columns=['Stories','homebath'], values=['homeprice'], aggfunc='mean')

# cross tab frequency of occurences
pd.crosstab(dframe.homebath, dframe.homebr, margins = True)

###############################################################
###															###
###															###
###   						GROUP BY						###
###															###
###															###
###############################################################

#groupby literally creates grouped objects in python. You can then perform actions on the groups

import os
os.chdir('C:\\Users\\Aroon\\Documents\\Kaggle')
dframe = pd.read_csv('WHO.csv')
dframe.dtypes #shows dtype for all columns in dframe

# simple example
grp_who = dframe['Population'].groupby(dframe['Region']) #groups population data based on Region
group_mean = dframe['Population'].groupby(dframe['Region']).mean() #or you can use max(), min(), sum()

#lets import some more data and do some high level discovery and manipulation
dframe = pd.read_csv('homedata3.csv')
dframe.shape # number of rows and columns
dframe.dtypes # list all column names
dframe.describe() # get some common stats
dframe.info() # gets the number of non null values across all columns
dframe.Stories = dframe.Stories.astype(str) # convert "Stories" to string
dframe.homebr = dframe.homebr.astype(str)
dframe.homebath = dframe.homebath.astype(str)

# more targetted group by statements

dframe.groupby(['zone','Stories']).mean() # returns mean value of all columns grouped by zone and Stories
dframe.groupby(['zone','Stories'])[['homeprice']].mean() #returns mean value of homeprice grouped by zone and stories

# group by using a user defined function
def my_range(arr):
	return arr.max() - arr.min()

home_grp = dframe.groupby('zone') 

home_grp.agg(my_range) #run UDF across all columns
home_grp['homeprice'].agg(my_range) # run UDF across homeprice data

###############################################################
###															###
###															###
###   						APPLY							###
###															###
###															###
###############################################################

# in this example we will rank house based on house price in groups based on zones

def rank(df):
	df['Grp_Rank'] = np.arange(len(df))+1
	return df
	
dframe.sort('homeprice', ascending = False, inplace = True)
dframe2 = dframe.groupby('zone').apply(rank)

dframe2[dframe2.Grp_Rank == 1] # returns the most expensive houses per group

# in the next example, we apply a function and create a new column for a dataframe

def br_bd(anumber):
    br,bd = anumber
    
    if br == '1':
        return 'studio'
    else:
        return bd
    
dframe['hometype'] = dframe[['homebr','homebath']].apply(br_bd, axis =1)

###############################################################
###															###
###															###
###   					Basic Analytics						###
###															###
###															###
###############################################################

'''
binning
'''
years = [1990,1991,1992,2008,2012,2015,1987,1969,2013,2008,1999] #some list with data
decade_bins = [1960,1970,1980,1990,2000,2010,2020] #decide on your bins
decade_cat = pd.cut(years,decade_bins) # pd.cut() bins the data. P/P of decade_cat is data element wise bin category
decade_cat.categories # O/P is bin categories
hist = pd.value_counts(decade_cat) # creates a count of data elements within a specific bin. O/P is series
pd.cut(years,4,precision = 1) # cuts the data into 4 equal spaced bins

''' 
starting analysis
'''

dframe.shape # number of rows and columns
dframe.dtypes # list all column names
dframe.describe() # get some common stats
dframe.info() # gets the number of non null values across all columns

'''
describe data. Quick stats on columns of dataframe
'''
np.random.seed(12345)
df = DataFrame(np.random.randn(1000,4))
df.describe() 
df.describe(['Column_you_want_to_focus_on'])

# you can also run describe on groups
home_grp = dframe.groupby('zone')
home_grp.describe()

# for other scalars you can use the following syntax
df['column_name'].mean() # you can use min, max, sum, etc
df['factor_column'].value_counts() # returns count of each factor in the column

'''
Managing outliers
'''
# lets first define and identify our outlier
outlier = 3
df[(np.abs(df)>outlier).any(1)] #look for all absolute (abs) values in all columns of the dataframe where any one [.any(1)] of the rows in the df exceeds the outlier
df[abs(df[0]) > outlier] # look at column [0] and return all rows where df[0] exceeds outlier
# now lets replace the outliers
df[np.abs(df)>3] = np.sign(df)*3 #np.sign(df) get the sign (+/-) of that elements. Here we cap all outliers exceeding limit 3 with the max value 3
df.ix[abs(df[0]) > outlier, [0]] = np.sign(df[0])*outlier #more realistic example of replacing outlier in a particular column with a limit value

'''
separating test and train data
'''
df = DataFrame(np.arange(160).reshape(40,4),columns = ['A','B','C','D'])
test = random.sample(df.index,long(round(0.2*len(df))))
testdata = df.ix[test]
traindata = df.drop(test)

'''
Moving averages
'''
# we use panda's builtin function rolling_mean()
ma_day = [50,100,200]

for ma in ma_day:
	col_name = "{0} days average".format(str(ma))
	FB[col_name] = pd.rolling_mean(FB['Adj Close'],ma)


###############################################################
###															###
###															###
###   					VISUALIZATION						###
###															###
###															###
###############################################################

from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid') #background is set to white
%matplotlib inline

''' basic plot '''
dframe['column'].plot(legend = True, figsize=(10,4))

sns.set_style('white')
FB[['50 days average','10 days average', '5 days average']].plot(legend = True, figsize=(15,8)) #multiple run charts

''' scatter plot '''
dframe.plot(kind = 'scatter', x='homedist', y='homeprice') #scatter plot
sns.jointplot(dframe.homedist, dframe.homeprice) #using seaborn. Also provides p-value and histogram
sns.jointplot(dframe.homedist, dframe.homeprice, kind = 'hex') # hex reveals density

''' histogram '''
bins = list(range(1,11))
plt.hist(dframe.homebr, bins = bins)

dframe['homeage'].hist(bins = 20)

#overlapping histograms
plt.hist(dframe.homebr, normed = True, alpha = 0.4, bins = bins, color = 'red') # normed normalizes the data to %ages
plt.hist(dframe.homebath, normed = True, alpha = 0.4, bins = bins, color = 'yellow')

''' Kernel density estimates '''
'''
Kernel density estimation is a fundamental data smoothing problem where inferences about the population are made
inorder to estimate the shape of this function kernel density estimator is used.
the kernel is a non-negative function that integrates to one and has mean zero
bandwidth is a smoothing parameter. Very low values results in undersmoothed curves since it contains too many spurious data artifacts
	very high values leads to oversmoothness and as a result obscures much of the underlying structure
	Think of bandwidth selection as a bias/variane tradeoff
'''
sns.kdeplot(dframe.homebr) # good old kde plot
sns.kdeplot(dframe.homeprice, cumulative = True) # cumulative distribution function (cdf)
sns.kdeplot(dframe.homeprice, dframe.homeage, shade = True) # multivariate distribution function. Remember mdf uses covariance matrix v/s variance for a univariate distribution function
sns.jointplot(dframe.homeprice, dframe.homeage, kind='kde')

''' Distplot ''' # bringing histograms and kde together
sns.distplot(dframe.homeprice)

sns.distplot(dframe.homeprice, 
            kde_kws={'color' : 'blue'},
            hist_kws = {'color': 'red'})
			
''' Box plot and Violin plots '''
#Box plot relies on range and median
#Violin plot embeds Kde within a BoxPlot revelaing the underlying distribution as well


sns.boxplot(x = dframe.zone, y = dframe.homeprice)
sns.violinplot(x = dframe.zone, y = dframe.homeprice)

''' Regression plots '''
tips = sns.load_dataset('tips') #load tips data
sns.lmplot(x='tip', y='total_bill',data = tips) #basic reg plot

sns.lmplot(x='tip', y='total_bill',data = tips,
           scatter_kws={'marker': 'x', 'color':'black'},
          line_kws = {'linewidth':0.5,'color':'orange'}) #few formatting options
		  
sns.lmplot(x='tip', y='total_bill',data = tips, order =2) #regression plot with 'total_bill'^2 (order = 2)

sns.lmplot(x="total_bill", y="tip", hue="smoker", markers = ['x','o'] ,data=tips) # separate lm plots for smoker and non-smoker
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, palette=dict(Yes="g", No="m"))

sns.lmplot(x="size", y="total_bill", hue="smoker", col="day", data=tips, aspect=.4, x_jitter=.1) #aspect is used for making the graphs more compact or large (if aspect >1)

sns.lmplot(x="total_bill", y="tip", col="day", hue="day", data=tips, col_wrap=2, size=3) #col_wrap = number of columns

sns.lmplot(x="total_bill", y="tip", row="sex", col="time", data=tips, size=3) # Condition on two variables sex and time to make a full grid

sns.lmplot(x='homebr',y='homeprice',data = dframe, x_estimator=np.mean) # x_estimator shd be used when 'x' is discrete. 

sns.lmplot(x='homedist',y='homeprice',hue = 'Stories', data = dframe) # when plotting 2 continuous variables, you get a lot of data points on your graph.
# to simplify the above visual, we use bins
pricebins = [10,20,30,40]
sns.lmplot(x='homedist',y='homeprice',hue = 'Stories', data = dframe, x_bins = pricebins)

# Regression plots along with other plots
fig, (axis1, axis2) = plt.subplots(1,2,sharey=True) #sharey=True ensures y axis on both plots are shared
sns.regplot('homeage','homeprice', dframe, ax = axis1)
sns.violinplot(x='homebr', y='homeprice', data = dframe, ax = axis2)

''' HEAT MAP and BAR PLOT '''

#load data
fdframe = sns.load_dataset('flights')
# format in the method suitable for heatmap
fdframe = fdframe.pivot('month','year','passengers')
# Visualize
sns.heatmap(fdframe)
sns.heatmap(fdframe, annot=True, fmt = 'd') # this embeds the numbers within heatmap
sns.heatmap(fdframe,center = fdframe.loc['January',1955]) # diverging visual

# heat map with a bar plot
fig, (axis1, axis2) = plt.subplots(2,1)

# the next few steps are taken to convert a series into a dataframe
year_series = fdframe.sum()

years = pd.Series(year_series.index.values)
years = pd.DataFrame(years)

flights = pd.Series(year_series.values)
flights = pd.DataFrame(flights)

year_dframe = pd.concat((years, flights), axis = 1)
year_dframe.columns = ['Year', 'Flights']

'''
the above 7 lines can be replaced with this code
fdframe2 = sns.load_dataset('flights')
fdf_grp = pd.DataFrame(fdframe2['passengers'].groupby(fdframe2['year']).sum())
fdf_grp.reset_index(level=0, inplace = True)
'''

sns.barplot(x='Year', y='Flights', data = year_dframe, ax = axis1)
sns.heatmap(fdframe, cmap = 'Blues', ax = axis2, cbar_kws = {'orientation':'horizontal'})

''' Factorplot. ''' # As the name suggests great plot for factor data
sns.factorplot(x='homebr', data = dframe, kind="count", hue = 'zone', aspect = 4, palette = 'Blues') #barchart (count)
sns.factorplot(x='homebr', y='homeprice',hue = 'Stories',data = dframe, aspect = 4) # homeprice runchart across 'homebr' (mean)
# kind : {point, bar, count, box, violin, strip}

''' FacetGrid. ''' # FacetGrid is used to draw plots with multiple Axes where each Axes shows the same relationship conditioned on different levels of some variable
myimg = sns.FacetGrid(dframe,hue = 'Stories', col = 'zone', row = 'homebr') # set the grid
myimg = myimg.map(sns.pointplot, 'pricePerSqft') # set the plot type

myimg = sns.FacetGrid(dframe,hue = 'Stories', row = 'zone', aspect = 4) # set the grid
myimg = myimg.map(sns.kdeplot, 'homeprice', shade = True).add_legend().set_axis_labels("Home Prices") # set the plot type

''' Correlation Visualization ''' # aka correlation plots
sns.pairplot(dlyReturns_df)

dlyReturns_fig = sns.PairGrid(dlyReturns_df, size = 5, aspect = 2)
dlyReturns_fig.map_upper(plt.scatter, color = 'darkblue')
dlyReturns_fig.map_lower(sns.kdeplot, cmap = 'cool_d')
dlyReturns_fig.map_diag(plt.hist, bins = 30)

sns.corrplot(dlyReturns_df, annot = True)

###############################################################
###															###
###															###
###   				Importing stock prices					###
###															###
###															###
###############################################################

import pandas.io.data as pdweb
import datetime
from pandas.io.data import DataReader
from datetime import datetime
from __future__ import division # dont have to worry about division complications with python 2.7

# One way of getting data

prices = pdweb.get_data_yahoo(['CVX','XOM','BP'],start = datetime.datetime(2011,1,1), 
                              end = datetime.datetime(2014,1,1))['Adj Close']
							  
volume = pdweb.get_data_yahoo(['CVX','XOM','BP'],start = datetime.datetime(2011,1,1), 
                              end = datetime.datetime(2014,1,1))['Volume']

rets = prices.pct_change() # returns percent change across columns. Row[1] - Row[0]/Row[0]

%matplotlib inline # plots within Jupyter notebook
prices.plot() # line plot

# A better more advanced way

my_list = ['FB', 'AMZN' , 'NFLX', 'GOOG']
end = datetime.now()
start = datetime(end.year-1, end.month, end.day)

for stock in my_list:
    globals()[stock] = DataReader(stock,'yahoo', start, end) #globals ensures that the variable is available outside the for loop
	
# if you just need one column ... say closing price (for example)
closing_df = DataReader(my_list,'yahoo', start, end)['Adj Close']
dlyReturns_df = closing_df.pct_change().dropna()

## correlations
corr = rets.corr() # returns corelation matrix between columns
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.corrplot(rets,annot = False,diag_names=False)