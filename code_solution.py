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
  

def matrix_invertible(candidate: np.array, col_index: int) -> bool:
  
    invertible = (np.all(candidate[:,col_index]) == 0)
    return invertible


def solve_zero_index(candidate: np.array, row_index: int, col_index: int) -> np.array:
  
  if not matrix_invertible(candidate, col_index):
    raise MatrixIsSingular

  if candidate[col_index, row_index] == 0:
    candidate_for_addition = find_candidate(candidate, row_index + 1, col_index)
    candidate[row_index] += candidate_for_addition
  
  candidate[row_index] /= candidate[row_index, col_index]
  return candidate

def reduce_to_0_echelon(row: np.array, index: int) -> np.array:
  for i in range(index):
    if row[i] > 0:
      row -= row[i]
    elif row[i] < 0:
      row+= (row[i] * -1)

    return row
    

def solve_non_zero_index(candidate: np.array, row_index: int, col_index: int) -> np.array:

  current_row = candidate[row_index]
  current_row = reduce_to_0_echelon(current_row, col_index)
  print(current_row)
  candidate[row_index] = current_row
  candidate[row_index] /= candidate[row_index, col_index]
  return candidate


A = np.array([
        [0, 7, -5, 3],
        [2, 8, 0, 4],
        [3, 12, 0, 5],
        [1, 3, 1, 3]
    ], dtype=np.float_)

print('GOING INTO FUNCTION =>')
print(A)

A = solve_zero_index(A, 0, 0)
print('0 Index Solved =>')
print(A)
print('1 Index Solved =>')
A = solve_non_zero_index(A, 1, 1)
print(A)