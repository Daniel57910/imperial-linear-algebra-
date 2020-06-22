import gran_schmidt
import numpy as np
from numpy.linalg import norm, inv
from numpy import transpose

def build_reflection_matrix(bearBasis) : 
    
    orthonormal_matrix = gran_schmidt.gsBasis(bearBasis)
    orthonormal_inverse = inv(orthonormal_matrix)
    transformation_matrix = np.array([[0, 1], [0, -1]])
    transformation_matrix = transformation_matrix @ orthonormal_inverse
    return transformation_matrix @ orthonormal_matrix

bearBasis = np.array(
    [[1,   -1],
     [1.5, 2]])
# This line uses your code to build a transformation matrix for us to use.
T = build_reflection_matrix(bearBasis)



