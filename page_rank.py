import numpy as np
import numpy.linalg as la


SITE_PROBABILITY_MATRIX = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
              [1/3, 0,   0,   0, 1/2, 0 ],
              [1/3, 1/2, 0,   1, 0,   0.5 ],
              [1/3, 0,   1/3, 0, 1/2, 0.5 ],
              [0,   0,   0,   0, 0,   0 ],
              [0,   0,   1/3, 0, 0,   0 ]]
              )

def calculate_page_rank_brute_force(site_probability_matrix: np.array) -> np.array:

    base_site_land_vector = np.ones(site_probability_matrix.shape[0]) / site_probability_matrix.shape[0]
    current_site_land_vector = site_probability_matrix @ base_site_land_vector
    while (la.norm(base_site_land_vector - current_site_land_vector) > 0.01):
        base_site_land_vector = current_site_land_vector
        current_site_land_vector = site_probability_matrix @ current_site_land_vector

    return current_site_land_vector, base_site_land_vector

def calculate_page_rank_eigen_decomposition(site_probability_matrix: np.array) -> np.array:
    print(site_probability_matrix)
    eigenvalues, eigenvectors = la.eig(site_probability_matrix)
    eigenvalues = np.diag(eigenvalues)

    result = eigenvalues @ la.inv(eigenvectors)
    return eigenvectors @ result

