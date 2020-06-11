import numpy as np
import copy
LIMIT = 4

B = np.array([
        [0, 7, -5, 3],
        [2, 8, 0, 4],
        [3, 12, 0, 5],
        [1, 3, 1, 3]
    ], dtype=np.float_)

A = np.array([
    [2, 0, 0, 0],
    [0, 3, 0, 0],
    [0, 0, 4, 4],
    [0, 0, 5, 5]
], dtype=np.float_)

def convert_to_echelon(matrix: np.array) -> np.array:

    for i in range(0, LIMIT):
        current_row = matrix[i]
        if current_row[i] == 0:
            matrix = swap(matrix, i)
        current_row = fix(matrix, current_row, i)
        matrix[i] = current_row
    
    return matrix
        

def swap(matrix: np.array, i: int) -> np.array:
    temp_row = copy.deepcopy(matrix[i])
    matrix[i] = matrix[i + 1]
    matrix[i + 1] = temp_row
    return matrix


def fix(matrix: np.array, current_row: np.array, i: int) -> np.array:
    for index in range(0, i):
        current_row -= current_row[index] * matrix[index]
    
    current_row /= current_row[i]
    return current_row


def in_echelon_form(A: np.array) -> np.array:
    
    A = removed_non_information_rows(A)
    axis_array = list(map(lambda row: np.where(row == 1)[0][0], A))

    if not axis_array == sorted(axis_array) or trailing_zero(axis_array, A):
        return False
    
    return True


def removed_non_information_rows(A):
    for i in range(0, LIMIT):
        if np.count_nonzero(A[i]) == 0:
            A = np.delete(A, i, axis=0)
    
    return A[~np.isnan(A).any(axis=1)]


def trailing_zero(axis_array: list, A: np.array) -> bool:

    for i in range(0, len(axis_array)):
        leading_1_index, row = axis_array[i], A[i]
        preceding_values = row[:leading_1_index]
        if np.count_nonzero(preceding_values) > 0:
            return True
    
    return False