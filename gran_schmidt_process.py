# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14  # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):    
    B = np.array(A, dtype=np.float_)
    B[0] = convert_row_to_length_1(B[0])

    for i in range(1, 2):
      current = B[i]
      vector_projection_spaces = []
      for j in range(0, i):
        vector_projection_spaces.append(get_projection_multiplier(current, B[j]))
      
      current -= np.sum(vector_projection_spaces)
      B[i] = convert_row_to_length_1(current)
    
    return B




def convert_row_to_length_1(row: np.array) -> np.array:
  return (row / la.norm(row))

def get_projection_multiplier(current: np.array, normalized_vector: np.array) -> int:
  return (current @ normalized_vector) * normalized_vector


V = np.array([[1, 0, 2, 6],
              [0, 1, 8, 2],
              [2, 8, 3, 1],
              [1, -6, 2, 3]], dtype=np.float_)

