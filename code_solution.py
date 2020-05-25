# GRADED FUNCTION
import numpy as np
LIMIT = 4
class MatrixIsSingular(Exception): 
  pass


A = np.array([
        [0, 7, -5, 3],
        [2, 8, 0, 4],
        [3, 12, 0, 5],
        [1, 3, 1, 3]
    ], dtype=np.float_)

print('GOING INTO FUNCTION =>')
print(A)

def find_candidate(candidate: np.array, row_index, col_index) -> np.array:

  for i in range(row_index, LIMIT):
    if candidate[i, col_index] != 0:
      return candidate[i]
  


def solve_zero_index(candidate: np.array, row_index: int, col_index: int) -> np.array:
  
  column_for_echelon = candidate[:,col_index]

  if np.all(column_for_echelon == 0):
    raise MatrixIsSingular

  if candidate[col_index, row_index] == 0:
    candidate_for_addition = find_candidate(candidate, row_index + 1, col_index)
    candidate[row_index] += candidate_for_addition
    candidate[row_index] /= candidate[row_index, col_index]
    print(candidate)

A = solve_zero_index(A, 0, 0)