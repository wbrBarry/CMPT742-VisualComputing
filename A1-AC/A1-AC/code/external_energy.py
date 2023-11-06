import numpy as np
import cv2

def line_energy(image):
    # Line energy (intensity of the image)
    return image.astype(float)

def edge_energy(image):
    # Edge energy (gradient magnitude)
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)
    return gradient_magnitude

def term_energy(image):
    # Term energy (curvature)
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    dxx = cv2.Sobel(dx, cv2.CV_64F, 1, 0, ksize=3)
    dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1, ksize=3)
    dxy = cv2.Sobel(dx, cv2.CV_64F, 0, 1, ksize=3)

    Cx = dx
    Cy = dy
    Cxx = dxx
    Cyy = dyy
    Cxy = dxy

    #print("Cx =", Cx)
    epsilon = 1e-5  # Small value to avoid division by zero
    denominator = np.sqrt(Cx**2 + Cy**2)
    denominator[denominator < epsilon] = epsilon  # Replace values less than epsilon with epsilon
    E_term = (Cxx * Cy**2 - 2 * Cxy * Cx * Cy + Cyy * Cx**2) / denominator**3
    #print("E_term =", E_term)
    return E_term

def external_energy(image, w_line, w_edge, w_term):
    # External energy (weighted sum of line, edge, and term energies)
    line = line_energy(image)
    edge = edge_energy(image)
    term = term_energy(image)

    external_energy = w_line * line + w_edge * edge + w_term * term

    return external_energy