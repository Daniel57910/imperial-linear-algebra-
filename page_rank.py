import numpy as np
import numpy.linalg as la
from functools import reduce


SITE_PROBABILITY_MATRIX = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   0.5 ],
              [1/3, 0,   1/3, 0, 1/2, 0.5 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]]
              )

SELF_LOOP_ARRAY = np.array(
    [[0,   1/2, 1/3, 0, 0,   0, 0 ],
     [1/3, 0,   0,   0, 1/2, 0, 0 ], 
     [1/3, 1/2, 0,   1, 0,   0, 0 ],
     [1/3, 0,   1/3, 0, 1/2, 0, 0 ],
     [0,   0,   0,   0, 0,   0, 0 ],
     [0,   0,   1/3, 0, 0,   0, 0 ],
     [0,   0,   0,   0, 0,   1, 1]]
)
def calculate_page_rank_brute_force(site_probability_matrix: np.array, count=100) -> np.array:

    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    current_site_land_vector = site_probability_matrix @ base_site_land_vector
    i = 0
    while (i < count):
        base_site_land_vector = current_site_land_vector
        current_site_land_vector = site_probability_matrix @ current_site_land_vector
        i+=1
    return current_site_land_vector, base_site_land_vector

def calculate_page_rank_eigen_decomposition(site_probability_matrix: np.array, count=100) -> np.array:
    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    eigenvalues, eigenvectors = la.eig(site_probability_matrix)
    diagonal_eigenvector_power = np.diag(eigenvalues) ** count
    result = diagonal_eigenvector_power @ la.inv(eigenvectors)
    transformation_matrix = eigenvectors @ result
    return transformation_matrix @ base_site_land_vector


def eigen_dampen(site_probability_matrix: np.array, count=100, dampen_parameter=0.3) -> np.array:
    site_probability_matrix = add_dampen_parameter(site_probability_matrix, dampen_parameter, site_probability_matrix.shape[0])
    
    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    eigenvalues, eigenvectors = la.eig(site_probability_matrix)
    diagonal_eigenvector_power = np.diag(eigenvalues) ** count
    result = diagonal_eigenvector_power @ la.inv(eigenvectors)
    transformation_matrix = eigenvectors @ result
    return transformation_matrix @ base_site_land_vector

def calculate_page_rank_brute_force(site_probability_matrix: np.array, count=47) -> np.array:

    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    current_site_land_vector = site_probability_matrix @ base_site_land_vector
    i = 0
    while (i < count):
        base_site_land_vector = current_site_land_vector
        current_site_land_vector = site_probability_matrix @ current_site_land_vector
        i+=1
    return current_site_land_vector, base_site_land_vector

def calculate_brute_force_dampen(site_probability_matrix: np.array, count=100, dampen_parameter=0.3) -> np.array:

    site_probability_matrix = add_dampen_parameter(site_probability_matrix, dampen_parameter, site_probability_matrix.shape[0])

    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    current_site_land_vector = site_probability_matrix @ base_site_land_vector
    i = 0
    while (i < count):
        base_site_land_vector = current_site_land_vector
        current_site_land_vector = site_probability_matrix @ current_site_land_vector
        i+=1
    return current_site_land_vector, base_site_land_vector


def add_dampen_parameter(site_probability_matrix: np.array, dampen_parameter: int, shape: int) -> np.array:
    site_probability_matrix *= dampen_parameter
    one_matrix = np.ones((shape, shape)) * (1 - dampen_parameter) / shape
    return site_probability_matrix + one_matrix

    