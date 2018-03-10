#!/usr/bin/env python3

import numpy as np

print("\nRow Vector")
row = np.array([[6, -1, 4]])
print(np.shape(row))

print("\nCol Vector")
col = row.T
print(np.shape(col))

print("\nCol Vector")
col = np.array([[2],[-1],[0]])
print(np.shape(col))

print("\nVector Operations")
row1 = np.array([[3,4,1]])
row2 = np.array([[4,-4,4]])

out = np.add(row1, row2)
print(out)

out = np.subtract(row1, row2)
print(out)

out = np.multiply(row1, row2)
print(out)

out = np.divide(row1, row2)
print(out)

out = np.dot(row1[0], row2[0])
print(out)

print("\nMatrix Operations")
mat1 = np.matrix([[3,1,5],[-4,1,2]]) # 2x3
mat2 = np.matrix([[3,2],[-5,0],[4,1]]) # 3x2

print(mat1.T)

out = np.add(mat1, mat2.T)
print(out)

out = np.matmul(mat1, mat2)
print(out)

# Dot product, same as np.matmul
out = np.dot(mat1, mat2) 
print(out)

print("\nGeneric Operations")
# Max of mat
out = np.maximum(0, mat1)
print(out)

# Max of row
out = np.maximum(0, row2)
print(out)

print("\nInitilizations")
# Zeros
out = np.zeros((2,3))
print(out)

# Random
out = np.random.rand(2, 3)
print(out)
