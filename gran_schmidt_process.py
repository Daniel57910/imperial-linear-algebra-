# GRADED FUNCTION
import numpy as np
import numpy.linalg as la

verySmallNumber = 1e-14  # That's 1×10⁻¹⁴ = 0.00000000000001

def gsBasis4(A):    
    B = np.array(A, dtype=np.float_)
  
    for i in range(0, 2):
      current = B[i]
      for j in range(0, i):
        orthonormal_compliment = get_projection_multiplier(current, B[j])
        current -= orthonormal_compliment
      B[i] = convert_row_to_length_1(current)
    
    return B




def convert_row_to_length_1(row: np.array) -> np.array:
  return row / la.norm(row)

# Length of normalized vector 1 so AB = AB / B**2
def get_projection_multiplier(current: np.array, normalized_vector: np.array) -> np.array:
  return (current @ normalized_vector) * normalized_vector


