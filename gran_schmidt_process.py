# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14  # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):    
    B = np.array(A, dtype=np.float_)
    B[0] = convert_row_to_length_1(B[0])
    for i in range(1, 4):
      for j in range(0, i):
        print(f'Vector {B[i]} -> {B[j]}')

def gsBasis(A):
    return B

def dimensions(A):
    return np.sum(la.norm(gsBasis(A), axis=0))

def convert_row_to_length_1(row: np.array) -> np.array:
  return (row / la.norm(row))



V = np.array([[1, 0, 2, 6],
              [0, 1, 8, 2],
              [2, 8, 3, 1],
              [1, -6, 2, 3]], dtype=np.float_)


gsBasis4(V)