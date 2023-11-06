import cv2
import numpy as np
from scipy.sparse import linalg, lil_matrix

from align_target import align_target

def poisson_blend(source_image, target_image, target_mask):
    # Convert the mask to bool type
    target_mask = target_mask.astype(bool)
    source_image = source_image.astype(np.float64)
    target_image = target_image.astype(np.float64)
    # Create an index matrix, and an array that contains the indices of the mask pixels
    idx = np.arange(target_image.shape[0] * target_image.shape[1]).reshape(target_image.shape[0], target_image.shape[1])
    mask_idx = idx[target_mask]

    # Calculate the total number of mask pixels
    n = mask_idx.size

    blended_image = target_image.copy()

    total_error = 0   # Initialize the total error

    for channel in range(3):  # Loop over each channel: R, G, B
        source_channel = source_image[:, :, channel]
        target_channel = target_image[:, :, channel]

        # Compute the Laplacian of the source image channel
        laplacian = -cv2.Laplacian(source_channel, cv2.CV_64F)
        laplacian_target = -cv2.Laplacian(target_channel, cv2.CV_64F)
        alpha = 0.75  # Adjust as needed
        laplacian = alpha * laplacian + (1 - alpha) * laplacian_target
        #normalized_laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        #cv2.imshow("Normalized Laplacian", normalized_laplacian)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        # Create the Poisson matrix A and the right-hand side vector b
        A = lil_matrix((n, n), dtype=float)
        b = np.zeros(n, dtype=float)

        for i, index in enumerate(mask_idx):
            A[i, i] = 4
            b[i] = laplacian[index // target_image.shape[1], index % target_image.shape[1]]

            neighbors = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # left, up, right, down
            for dx, dy in neighbors:
                y, x = index // target_image.shape[1] + dy, index % target_image.shape[1] + dx
                if 0 <= x < target_image.shape[1] and 0 <= y < target_image.shape[0]:
                    if target_mask[y, x]:  # If the neighbor is inside the mask
                        A[i, np.where(mask_idx == y * target_image.shape[1] + x)[0][0]] = -1
                    else:  # If the neighbor is outside the mask
                        b[i] += target_channel[y, x]

        # Solve the linear system Ax = b
        x = linalg.spsolve(A.tocsr(), b)
        #print(x)

        error_channel = np.linalg.norm(A.dot(x) - b)   # Compute the error for the current channel
        total_error += error_channel   # Update the total error

        # Map the result to the target channel
        for i, value in enumerate(x):
            blended_value = np.clip(value, 0, 255)
            blended_image[mask_idx[i] // blended_image.shape[1], mask_idx[i] % blended_image.shape[1], channel] = blended_value

        #print("Blending values:", x.min(), x.max(), x.mean())

    print("Total Least Squares Error:", total_error)   # Print the total error after channel loop ends

    return blended_image

if __name__ == '__main__':
    #read source and target images
    source_path = 'source2.jpg'
    target_path = 'target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)
    
    # Apply Gaussian Blur to the mask to create a feathered effect
    feathered_mask = cv2.GaussianBlur(mask, (5, 5), 0)
    feathered_mask = feathered_mask / 255.0   # Normalize the mask values to [0, 1]
    
    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)

    cv2.imshow('Blended Image', blended_image.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
