import math
from math import sqrt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
import imutils
# from skimage import exposure
import argparse
import random


def order_corner_points(corners):
  # Separate corners into individual points
  # Index 0 - top-right
  #       1 - top-left
  #       2 - bottom-left
  #       3 - bottom-right
  corners = [(corner[0][0], corner[0][1]) for corner in corners]
  top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
  return (top_l, top_r, bottom_r, bottom_l)

def perspective_transform(image, corners):
  # Order points in clockwise order
  top_l, top_r, bottom_r, bottom_l = corners

  # Determine width of new image which is the max distance between
  # (bottom right and bottom left) or (top right and top left) x-coordinates
  width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
  width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
  width = max(int(width_A), int(width_B))

  # Determine height of new image which is the max distance between
  # (top right and bottom right) or (top left and bottom left) y-coordinates
  height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
  height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
  height = max(int(height_A), int(height_B))

  # Construct new points to obtain top-down view of image in
  # top_r, top_l, bottom_l, bottom_r order
  dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                  [0, height - 1]], dtype = "float32")

  # Convert to Numpy format
  ordered_corners = np.array(corners, dtype="float32")

  # Find perspective transform matrix
  matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

  # Return the transformed image
  return cv2.warpPerspective(image, matrix, (width, height))



CHESSBOARD_FILE_PATH = 'imgs/photo7.jpg'
PADDING = (15, 20)
NEED_ROTATE = True
NEED_PADDING = True
OUTPUT_IMAGE_SIZE = (500, 500)

image = cv2.imread(CHESSBOARD_FILE_PATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(gray, (9, 9), 0)

thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

thresh = cv2.bitwise_not(thresh)

kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
thresh = cv2.dilate(thresh, kernel, iterations=1)

# find biggest square area
cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse = True)

peri = cv2.arcLength(cnts[0], True)
biggest_cnt = cv2.approxPolyDP(cnts[0], 0.025 * peri, True)

imagedrawed = image.copy()
imagedrawed = cv2.drawContours(imagedrawed, [biggest_cnt], -1, (255,0,0), 5)

main_corners = order_corner_points(biggest_cnt)

transformed = perspective_transform(image.copy(), main_corners)
# cv2.imwrite('transformed.jpg', transformed)
cv2.imshow('gdsgds', imagedrawed)
cv2.waitKey(0)
cv2.destroyAllWindows()


SQUARE_SIZE = 3*8  # in cm (or any consistent unit)

# Prepare object points: (0,0,0), (1,0,0), ..., (8,5,0) * square size
# objp = np.zeros((, 3), np.float32)
# objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = np.array([[0, 0, 0],
                 [1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0]], dtype=np.float32)
objp *= SQUARE_SIZE
main_corners = np.array(main_corners).astype(np.float32).reshape(-1, 1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
corners_refined = cv2.cornerSubPix(gray, main_corners, (11, 11), (-1, -1), criteria)

# Dummy camera intrinsics for example purposes
# Replace with your calibrated camera matrix and distortion coefficients
h, w = gray.shape
camera_matrix = np.array([[1566, 0, w / 2],
                      [0, 1575, h / 2],
                      [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.zeros((5, 1))  # assume no distortion for simplicity

# Solve for rotation and translation vectors
ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

if ret:
    print("rvec (rotation vector):\n", rvec)
    print("tvec (translation vector):\n", tvec)
    print(corners_refined)
else:
    print("solvePnP failed to find a solution.")

# dist_coeffs = np.zeros((5, 1))
#
# # Draw axis on image
# axis_length = 3  # mm
# axis = np.float32([[axis_length, 0, 0],   # X - red
#                    [0, axis_length, 0],   # Y - green
#                    [0, 0, -axis_length]]) # Z - blue (into the board)
#
# imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
# corner = tuple(corners_refined[0].ravel().astype(int))
# image = cv2.line(image, corner, tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)  # X - red
# image = cv2.line(image, corner, tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)  # Y - green
# image = cv2.line(image, corner, tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)  # Z - blue
# for corner in corners_refined:
#     x, y = corner.ravel()
#     cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)  # Red circles for refined corners
#
# # Display the image with detected and refined corners
# cv2.imshow('Corners Visualization', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Measured real-world position of a point on the checkerboard (e.g., bottom-left corner)
world_point_pose = np.array([0, 0, 0])  # in meters

# Scale to mm for consistent units with tvecs
world_point_pose_mm = world_point_pose * 1000

# === Helper: Convert rvec, tvec to 4x4 transformation matrix ===
def matrix_from_vecs(tvec, rvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

# === Step 1: Pose of board in camera frame ===
board_pose_in_camera_frame = matrix_from_vecs(tvec, rvec)

# === Step 2: Pose of camera in board frame ===
pose_of_camera_in_board_frame = np.linalg.inv(board_pose_in_camera_frame)

# === Step 3: Transform from board to world ===
# Assuming world and board frames are aligned (i.e., no rotation), just translation
transform_from_board_to_world = np.eye(4)
transform_from_board_to_world[:3, 3] = world_point_pose_mm

# === Step 4: Camera pose in world frame ===
camera_in_world = np.dot(transform_from_board_to_world, pose_of_camera_in_board_frame)

# === Result ===
print("Camera position in world frame (mm):", camera_in_world[:3, 3])