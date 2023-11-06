import cv2
import numpy as np
from scipy.interpolate import CubicSpline

from external_energy import external_energy
from internal_energy_matrix import get_matrix

def interpolate_points(xs, ys, n):
    """
    Interpolates n points between the given set of points (xs, ys).
    """
    # Calculate the cumulative distance along the points
    dists = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    cum_dists = np.concatenate(([0], np.cumsum(dists)))
    total_dist = cum_dists[-1]

    # Generate n evenly spaced points along the distance
    new_dists = np.linspace(0, total_dist, n)

    # Create cubic spline for both x and y
    cs_x = CubicSpline(cum_dists, xs)
    cs_y = CubicSpline(cum_dists, ys)

    # Interpolate x and y values for these distances
    new_xs = cs_x(new_dists)
    new_ys = cs_y(new_dists)

    return new_xs, new_ys

def bilinear_interpolation(image, x, y):
    h, w = image.shape[:2]
    # find the nearest integer coordinates and increments them by 1 to find the other set of coordinates
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    # ensure they remain within image boundaries
    x0 = np.clip(x0, 0, w-1)
    x1 = np.clip(x1, 0, w-1)
    y0 = np.clip(y0, 0, h-1)
    y1 = np.clip(y1, 0, h-1)

    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa*Ia + wb*Ib + wc*Ic + wd*Id

def compute_gradients(image):
    # Assuming image is the external energy E
    fx = np.zeros_like(image)
    fy = np.zeros_like(image)

    # Compute gradients
    fx[:, :-1] = np.diff(image, axis=1)
    fy[:-1, :] = np.diff(image, axis=0)

    return fx, fy

def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        #save point
        xs.append(x)
        ys.append(y)

        #display point
        cv2.circle(img, (x, y), 3, 128, -1)
        cv2.imshow('image', img)

if __name__ == '__main__':
    #point initialization
    img_path = 'images/shape.png'
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_copy = np.copy(img)
    #print("image is:", img)
    # Blur the image for better results
    img_copy = cv2.GaussianBlur(img_copy, (5, 5), 0)
    #print("image after blur is:", img_copy)

    xs = []
    ys = []
    cv2.imshow('image', img)

    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #selected points are in xs and ys
    #print("xs:", xs)
    #print("ys:", ys)
    num_interpolated_points = 500  # Number of interpolated points
    # Interpolate between the selected points
    interpolated_xs, interpolated_ys = interpolate_points(xs, ys, num_interpolated_points)
    #print("interpolated xs:", interpolated_xs)
    #print("interpolated ys:", interpolated_ys)
    alpha = 0.06
    beta = 0.6
    gamma = 1
    kappa = 2
    num_points = num_interpolated_points  # Use the number of interpolated points

    M = get_matrix(alpha, beta, gamma, num_points)
    #print("M is:", M)
    w_line = 0.001
    w_edge = 1.5
    w_term = 0.5
    E = external_energy(img_copy, w_line, w_edge, w_term)
    #print("E is:", E)
    fx_grid, fy_grid = compute_gradients(E)

    for iteration in range(100):  # number of iterations
        # Initialize arrays to store the updated x and y coordinates
        new_xs = np.zeros(num_points)
        new_ys = np.zeros(num_points)
        fx = 0
        fy = 0

        for i in range(num_points):
            x = interpolated_xs[i]
            y = interpolated_ys[i]

            fx = bilinear_interpolation(fx_grid, x, y)
            fy = bilinear_interpolation(fy_grid, x, y)

            # Ensure no division by zero
            epsilon = 1e-5
            denominator = max(epsilon, np.sqrt(fx**2 + fy**2))

            new_xs[i] = M[i, :] @ (gamma * interpolated_xs - kappa * fx / denominator)
            new_ys[i] = M[i, :] @ (gamma * interpolated_ys - kappa * fy / denominator)

        interpolated_xs, interpolated_ys = new_xs, new_ys
    #print("range")
    # Draw contours on the image
    for i in range(num_points):
        cv2.circle(img_copy, (int(interpolated_xs[i]), int(interpolated_ys[i])), 3, 128, -1)

    # Display the final result
    cv2.imshow('Final Contour', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
