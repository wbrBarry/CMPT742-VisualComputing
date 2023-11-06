import numpy as np

"""Return the matrix for the internal energy minimization.
# Arguments
    alpha: The alpha parameter.
    beta: The beta parameter.
    gamma: The gamma parameter.
    num_points: The number of points in the curve.
# Returns
    The matrix for the internal energy minimization. (i.e. A + gamma * I)
"""

def get_matrix(alpha, beta, gamma, num_points):
    # Initialize a pentadiagonal matrix A of size num_points x num_points
    A = np.zeros((num_points, num_points))

    # Coefficients based on alpha and beta
    a = beta
    b = -(alpha + 4 * beta)
    c = 2 * alpha + 6 * beta

    # Fill the matrix A
    for i in range(num_points):
        A[i, i] = c # The main diagonal is filled with c
        A[i, (i-1) % num_points] = A[i, (i+1) % num_points] = b # The two diagonals immediately above and below the main diagonal are filled with b
        A[i, (i-2) % num_points] = A[i, (i+2) % num_points] = a # The diagonals representing the topmost and bottommost rows (to account for the periodic nature of the snake) are filled with a

    # Add gamma * I to A
    A += gamma * np.identity(num_points)

    # Compute the matrix M = (A + gamma * I)^-1
    try:
        M = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")

    return M