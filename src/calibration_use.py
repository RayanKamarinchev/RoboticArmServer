import cv2
import numpy as np

# Load image
img = cv2.imread("src/examples/latest (18).jpg")
# img = cv2.imread("src/calibration/capture (12).jpg")

# Your calibration results:
cameraMatrix = np.load("src/camera_matrix.npy")
distCoeffs   = np.load("src/dist_coeffs.npy")

# Undistort
undistorted = cv2.undistort(img, cameraMatrix, distCoeffs)

undistorted = cv2.rotate(undistorted, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("undistorted.jpg", undistorted)