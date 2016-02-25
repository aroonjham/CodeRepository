#Starting slow
#lets create an array and do scalar vector multiplication
import numpy as np

# Create a numpy array with the values 1, 2, 3
simpleArray = np.array([1, 2, 3])
# Perform the scalar product of 5 and the numpy array
timesFive = 5 * simpleArray
print simpleArray
print timesFive

# element wise multiplication and dot product
# Create a ndarray based on a range and step size.
u = np.arange(0, 5, .5)
v = np.arange(5, 10, .5)

elementWise = u*v
dotProduct = np.dot(u,v)
print 'u: {0}'.format(u) 
print 'v: {0}'.format(v)
print '\nelementWise\n{0}'.format(elementWise) # returns [  0.     2.75   6.     9.75  14.    18.75  24.    29.75  36.    42.75]
print '\ndotProduct\n{0}'.format(dotProduct) # returns 183.75

# Some matrix multiplication

from numpy.linalg import inv

A = np.matrix([[1,2,3,4],[5,6,7,8]]) #np.matrix() to generate a NumPy matrix
print 'A:\n{0}'.format(A)
# Print A transpose
print '\nA transpose:\n{0}'.format(A.T) # matrix.T creates transpose. In this case A.T

# Multiply A by A transpose
AAt = A*A.T
print '\nAAt:\n{0}'.format(AAt)

# Invert AAt with np.linalg.inv()
AAtInv = np.linalg.inv(AAt)
print '\nAAtInv:\n{0}'.format(AAtInv)

# Show inverse times matrix equals identity
# We round due to numerical precision
print '\nAAtInv * AAt:\n{0}'.format((AAtInv * AAt).round(4))

# Stacking arrays row wise and column wise

zeros = np.zeros(8) # returns an array of 8 0s [ 0.  0.  0.  0.  0.  0.  0.  0.]
ones = np.ones(8) # returns an array of 8 1s [ 1.  1.  1.  1.  1.  1.  1.  1.]
print 'zeros:\n{0}'.format(zeros)
print '\nones:\n{0}'.format(ones)

zerosThenOnes = np.hstack((zeros,ones))   #notice the "(("
# hstack will return [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.]
zerosAboveOnes = np.vstack((zeros,ones))   # A 2 by 8 array 
# vstack in the above example will return 	[[ 0.  0.  0.  0.  0.  0.  0.  0.]
# 											[ 1.  1.  1.  1.  1.  1.  1.  1.]]

print '\nzerosThenOnes:\n{0}'.format(zerosThenOnes)
print '\nzerosAboveOnes:\n{0}'.format(zerosAboveOnes)

# When using PySpark, we use DenseVector instead of numpy vector. Example below:

from pyspark.mllib.linalg import DenseVector

numpyVector = np.array([-3, -4, 5])
print '\nnumpyVector:\n{0}'.format(numpyVector)

# Create a DenseVector consisting of the values [3.0, 4.0, 5.0]
myDenseVector = DenseVector([3.0, 4.0, 5.0])
# Calculate the dot product between the two vectors.
denseDotProduct = DenseVector.dot(myDenseVector, numpyVector) # DenseVector.dot() does the dot product

print 'myDenseVector:\n{0}'.format(myDenseVector)
print '\ndenseDotProduct:\n{0}'.format(denseDotProduct)

