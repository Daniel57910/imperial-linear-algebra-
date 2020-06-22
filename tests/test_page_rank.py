from page_rank import calculate_page_rank_brute_force, calculate_page_rank_eigen_decomposition
import numpy as np
import numpy.linalg as la
from page_rank import SITE_PROBABILITY_MATRIX
import unittest

class TestPageRank(unittest.TestCase):

    def test_page_rank_on_simple_probability_matrix(self):
        site_probability_vector, previous_site_vector = calculate_page_rank_brute_force(
            np.array(SITE_PROBABILITY_MATRIX)
        )
        self.assertAlmostEqual(np.sum(site_probability_vector), 1)
        self.assertLessEqual(la.norm(site_probability_vector - previous_site_vector), 0.01)
    
    def test_eigenvector_decomposition(self):
        power_1_prob_matrix = calculate_page_rank_eigen_decomposition(
            np.array(SITE_PROBABILITY_MATRIX)
        )
        self.assertAlmostEqual(la.norm(power_1_prob_matrix), la.norm(SITE_PROBABILITY_MATRIX))