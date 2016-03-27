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

arr = np.arange(50).reshape([10,5]) #returns a 10x5 array
arr.T # transpose 
np.dot(arr,arr.T) # matrix multiplication

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