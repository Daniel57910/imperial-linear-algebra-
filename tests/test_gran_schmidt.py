import unittest
import gran_schmidt_process
import copy
import numpy as np
import numpy.linalg as la

V = np.array([[1, 0, 2, 6],
              [0, 1, 8, 2],
              [2, 8, 3, 1],
              [1, -6, 2, 3]], dtype=np.float_)


class TestEchelonCalc(unittest.TestCase):

  def test_convert_row_to_basis_set(self):

    test_row = V[0]
    test_row = gran_schmidt_process.convert_row_to_length_1(test_row)
    self.assertEqual(np.sum(la.norm(test_row)), 1)

  
  def test_gran_schmidt_on_length_2_vector(self):
    rows = V[0: 2]
    test = gran_schmidt_process.gsBasis4(rows)
    row_1, row_2 = test[0], test[1]
    self.assertEqual(np.sum(la.norm(row_1)), 1)
    self.assertEqual(np.sum(la.norm(row_2)), 1)
    self.assertEqual(row_1 @ row_2, 0)
