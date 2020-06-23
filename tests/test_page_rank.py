import page_rank
import numpy as np
import numpy.linalg as la
import unittest
import copy

class TestPageRank(unittest.TestCase):

    def test_page_rank_on_simple_probability_matrix(self):
        site_probability_vector, previous_site_vector = page_rank.calculate_page_rank_brute_force(
            np.array(page_rank.SITE_PROBABILITY_MATRIX)
        )
        self.assertAlmostEqual(np.sum(site_probability_vector), 1)
        self.assertLessEqual(la.norm(site_probability_vector - previous_site_vector), 0.01)
    
    def test_eigenvector_decomposition(self):
        # proof M**N*V = IE**N*I^-1V
        eigen_probability_vector = page_rank.calculate_page_rank_eigen_decomposition(
            np.array(page_rank.SITE_PROBABILITY_MATRIX)
        )
        site_probability_vector, previous_site_vector = page_rank.calculate_page_rank_brute_force(
            np.array(page_rank.SITE_PROBABILITY_MATRIX)
        )

        self.assertAlmostEqual(la.norm(eigen_probability_vector), la.norm(site_probability_vector))
    
    def test_brute_force_with_damping_parameter(self):
        current, base = page_rank.calculate_brute_force_dampen(np.array(page_rank.SELF_LOOP_ARRAY))
        self.assertAlmostEqual(np.sum(current), 1)
    
    def test_eigen_decomposition_with_damping_parameter(self):
        current, base = page_rank.calculate_brute_force_dampen(np.array(page_rank.SELF_LOOP_ARRAY))
        dampen_vector = page_rank.eigen_dampen(np.array(page_rank.SELF_LOOP_ARRAY))
        dampen_vector_sum = np.sum(dampen_vector)
        self.assertAlmostEqual(float(dampen_vector_sum), 1)
        for i in range(0, dampen_vector.shape[0]):
            self.assertAlmostEqual(current[i], dampen_vector[i])