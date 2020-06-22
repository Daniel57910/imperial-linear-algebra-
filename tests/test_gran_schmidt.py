import unittest
import gran_schmidt

import copy
import numpy as np
import numpy.linalg as la
from functools import reduce
V = np.array([[1, 0, 2, 6],
              [0, 1, 8, 2],
              [2, 8, 3, 1],
              [1, -6, 2, 3]], dtype=np.float_)

larger_vector_space = np.array([
    [1, 1, 2, 0, 1, 1],
    [0, 0, 0, 1, 2, 1],
    [1, 2, 3, 1, 3, 2],
    [1, 0, 1, 0, 1, 1]
], dtype=float)

class TestEchelonCalc(unittest.TestCase):

  def test_convert_row_to_basis_set(self):
    test_row = V[0]
    test_row = gran_schmidt.convert_row_to_length_1(test_row)
    self.assertEqual(np.sum(la.norm(test_row)), 1)

  
  def test_gran_schmidt_on_length_4_hardcoded(self):
    test = gran_schmidt.gsBasis4(V)
    for row in test:
      self.assertAlmostEqual(np.sum(la.norm(row)), 1)
    dot_product = reduce(np.dot, test)
    self.assertAlmostEqual(dot_product, 0)


  def test_gran_schmidt_on_length_n(self):
    test = gran_schmidt.gsBasis(V)
    for row in test:
      self.assertAlmostEqual(np.sum(la.norm(row)), 1)
    dot_product = reduce(np.dot, test)
    self.assertAlmostEqual(dot_product, 0)


  def test_on_larger_space(self):
    vector = gran_schmidt.gsBasis(larger_vector_space)
    for row in vector:
      self.assertAlmostEqual(np.sum(la.norm(row)), 1)
    dot_product = reduce(np.dot, vector)
    self.assertAlmostEqual(dot_product, 0)