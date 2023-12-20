import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

def find_matching_keypoints(image1, image2):
    #Input: two images (numpy arrays)
    #Output: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = []
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2

def drawlines(img1,img2,lines,pts1,pts2):
    #img1: image on which we draw the epilines for the points in img2
    #lines: corresponding epilines
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def normalize_points(pts):
    # Calculate the centroid of the points
    centroid = np.mean(pts, axis=0)

    # Compute the mean distance of points from the centroid
    mean_dist = np.mean(np.sqrt(np.sum((pts - centroid)**2, axis=1)))

    # Construct the normalization matrix
    scale = np.sqrt(2) / mean_dist
    T = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])

    # Normalize the points by converting them to homogeneous coordinates and applying T
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_normalized = np.dot(T, pts_homogeneous.T).T

    return pts_normalized[:, :2], T  # Return only the x, y coordinates

def FindFundamentalMatrix(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))
    # Normalize the points
    pts1_normalized, T1 = normalize_points(pts1)
    pts2_normalized, T2 = normalize_points(pts2)

    A = np.zeros((len(pts1), 9))
    for i, (p1, p2) in enumerate(zip(pts1_normalized, pts2_normalized)):
        x, y = p1
        x_prime, y_prime = p2
        A[i] = [x_prime * x, x_prime * y, x_prime, y_prime * x, y_prime * y, y_prime, x, y, 1]

    # Compute the Fundamental Matrix using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce the rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))

    # Denormalize the fundamental matrix
    F = np.dot(T2.T, np.dot(F, T1))

    # Normalize the scale of F so that the last element is 1
    F = F / F[2, 2]

    return F

def FindFundamentalMatrixRansac(pts1, pts2, num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))
    best_F = None
    max_inliers = 0

    for _ in range(num_trials):
        # Randomly sample 8 points
        indices = np.random.choice(len(pts1), 8, replace=False)
        sampled_pts1 = pts1[indices]
        sampled_pts2 = pts2[indices]

        # Compute the fundamental matrix using these 8 points
        F = FindFundamentalMatrix(sampled_pts1, sampled_pts2)  

        # Count the number of inliers
        inliers = 0
        for p1, p2 in zip(pts1, pts2):
            # Homogeneous coordinates
            p1_h = np.append(p1, 1)
            p2_h = np.append(p2, 1)

            # Check if the point pair is an inlier
            if np.abs(np.dot(p2_h.T, np.dot(F, p1_h))) < threshold:
                inliers += 1

        # Update the best fundamental matrix if needed
        if inliers > max_inliers:
            best_F = F
            max_inliers = inliers

    return best_F

if __name__ == '__main__':
    #Set parameters
    data_path = './data'
    use_ransac = False

    #Load images
    image1_path = os.path.join(data_path, 'mount_rushmore_1.jpg')
    image2_path = os.path.join(data_path, 'mount_rushmore_2.jpg')
    image1 = np.array(Image.open(image1_path).convert('L'))
    image2 = np.array(Image.open(image2_path).convert('L'))


    #Find matching keypoints
    pts1, pts2 = find_matching_keypoints(image1, image2)

    #Builtin opencv function for comparison
    F_true = cv2.findFundamentalMat(pts1, pts2, cv2.FM_8POINT)[0]
    print("Fundamental Matrix computed by built-in OpenCV function:", F_true)

    #todo: FindFundamentalMatrix
    if use_ransac:
        F = FindFundamentalMatrixRansac(pts1, pts2)
    else:
        F = FindFundamentalMatrix(pts1, pts2)
    print("My computed Fundamental Matrix:", F)

    # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1, img2 = drawlines(image1, image2, lines1, pts1, pts2)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()


    # Find epilines corresponding to points in first image, and draw the lines on second image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img1, img2 = drawlines(image2, image1, lines2, pts2, pts1)
    fig, axis = plt.subplots(1, 2)

    axis[0].imshow(img1)
    axis[0].set_title('Image 1')
    axis[0].axis('off')
    axis[1].imshow(img2)
    axis[1].set_title('Image 2')
    axis[1].axis('off')

    plt.show()