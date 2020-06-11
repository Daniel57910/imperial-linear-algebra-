# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14  # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):    
    B = np.array(A, dtype=np.float_)
    for i in range(0, 4):
      current = B[i]
      for j in range(0, i):
        current -= get_orthonormal_compliment(current, B[j])
      B[i] = convert_row_to_length_1(current)
    
    return B


def gsBasis(A):
    B = np.array(A, dtype=np.float_)
    row_count = np.shape(B)[0]
    for i in range(0, row_count):
      current = B[i]
      for j in range(0, i):
        current -= get_orthonormal_compliment(current, B[j])
      B[i] = convert_row_to_length_1(current)

    return B

def convert_row_to_length_1(row: np.array) -> np.array:
  return row / la.norm(row)

# Length of normalized vector 1 so AB = AB / B**2
def get_orthonormal_compliment(current: np.array, normalized_vector: np.array) -> np.array:
  return (current @ normalized_vector) * normalized_vector

def dimensions(A) :
    return np.sum(la.norm(gsBasis(A), axis=0))