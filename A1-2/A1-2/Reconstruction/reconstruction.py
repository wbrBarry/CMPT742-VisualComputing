'''
import numpy as np
import cv2
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def reconstruct_image(S, V, c1, c2, c3, c4):
    # Define the shape of the image
    n = S.shape[0]

    # Create the A matrix and b vector
    A = lil_matrix((n * n, n * n))
    b = np.zeros((n * n, 1))

    # Implement the reconstruction logic as described in the assignment
    for i in range(n):
        for j in range(n):
            index = i * n + j
            A[index, index] = 4

            b[index] = 4 * V[i, j] - V[max(i - 1, 0), j] - V[min(i + 1, n - 1), j] - V[i, max(j - 1, 0)] - V[i, min(j + 1, n - 1)]

            if i == 0:
                b[index] -= V[0, j]
            if i == n - 1:
                b[index] -= V[n - 1, j]
            if j == 0:
                b[index] -= V[i, 0]
            if j == n - 1:
                b[index] -= V[i, n - 1]

    # Add the constraints
    A[0, 0] = 1
    A[n - 1, n - 1] = 1
    A[(n - 1) * n, (n - 1) * n] = 1
    A[n * n - 1, n * n - 1] = 1

    b[0] = c1
    b[n - 1] = c2
    b[(n - 1) * n] = c3
    b[n * n - 1] = c4

    # Convert A to a CSR matrix for efficient computation
    A = A.tocsr()

    # Solve the system of equations
    v = spsolve(A, b)

    # Reshape the result to form the reconstructed image
    reconstructed_image = np.reshape(v, (n, n))

    # Compute the least square error |Av - b|
    reconstructed_flat = np.reshape(reconstructed_image, (n * n, 1))
    error = np.linalg.norm(A.dot(reconstructed_flat) - b)
    print(f"Least square error: {error}")

    return reconstructed_image


if __name__ == '__main__':
    # Read source image and target image
    S = cv2.imread('large.jpg', cv2.IMREAD_GRAYSCALE)  # Read source image as grayscale
    V = cv2.imread('large1.jpg', cv2.IMREAD_GRAYSCALE)  # Read target image as grayscale

    # Define constants c1, c2, c3, c4
    c1, c2, c3, c4 = 0, 0, 0, 0

    # Perform image reconstruction
    reconstructed_image = reconstruct_image(S, V, c1, c2, c3, c4)

    # Visualize the reconstructed image
    cv2.imshow('Reconstructed Image', reconstructed_image)
    cv2.waitKey(0)
'''
import numpy as np
import cv2
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import lsqr

def compute_derivatives(image):
    height, width = image.shape
    Sup = np.zeros_like(image)
    Sdown = np.zeros_like(image)
    Sleft = np.zeros_like(image)
    Sright = np.zeros_like(image)

    Sup[:-1, :] = image[:-1, :] - image[1:, :]
    Sdown[1:, :] = image[1:, :] - image[:-1, :]
    Sleft[:, :-1] = image[:, :-1] - image[:, 1:]
    Sright[:, 1:] = image[:, 1:] - image[:, :-1]

    return Sup, Sdown, Sleft, Sright

def build_matrix_and_vector(image, Sup, Sdown, Sleft, Sright):
    height, width = image.shape
    k = width * height
    A = lil_matrix((k, k))
    b = np.zeros(k)

    # Use original corner values
    c1, c2, c3, c4 = image[0, 0], image[height-1, 0], image[0, width-1], image[height-1, width-1]
    
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            
            # Set constraints for corners
            if (x, y) == (0, 0):
                A[idx, idx] = 1
                b[idx] = c1
                continue
            elif (x, y) == (0, height - 1):
                A[idx, idx] = 1
                b[idx] = c2
                continue
            elif (x, y) == (width - 1, 0):
                A[idx, idx] = 1
                b[idx] = c3
                continue
            elif (x, y) == (width - 1, height - 1):
                A[idx, idx] = 1
                b[idx] = c4
                continue
            
            # Else, handle edges and inner pixels
            coefficients = 0  # To count the number of coefficients added for v(x, y)

            if x > 0:
                A[idx, idx-1] = -1
                coefficients += 1
                b[idx] += Sleft[y, x]

            if x < width - 1:
                A[idx, idx+1] = -1
                coefficients += 1
                b[idx] += Sright[y, x]

            if y > 0:
                A[idx, idx-width] = -1
                coefficients += 1
                b[idx] += Sup[y, x]

            if y < height - 1:
                A[idx, idx+width] = -1
                coefficients += 1
                b[idx] += Sdown[y, x]
            
            # Set the coefficient for the center pixel
            A[idx, idx] = coefficients
                
    return A.tocsr(), b

def reconstruct_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(float) / 255.0
    
    Sup, Sdown, Sleft, Sright = compute_derivatives(image)
    A, b = build_matrix_and_vector(image, Sup, Sdown, Sleft, Sright)
    
    solution = lsqr(A, b)
    v = solution[0]
    reconstructed = v.reshape(image.shape)
    
    # Clamp the values and convert for visualization
    reconstructed = np.clip(reconstructed, 0, 1)
    reconstructed_image = (reconstructed * 255).astype(np.uint8)

    error = np.linalg.norm(A.dot(v) - b)
    
    return reconstructed_image, error

if __name__ == '__main__':
    image_path = "large.jpg"  # Change to the path of your image
    reconstructed_image, error = reconstruct_image(image_path)
    
    print(f"Least Square Error: {error}")
    
    cv2.imshow('Reconstructed Image', reconstructed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
