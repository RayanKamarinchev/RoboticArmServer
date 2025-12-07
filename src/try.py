import cv2 as cv
import numpy as np
from camera_utils import detect_aruco, get_camera_position, undistort_image, decoede_image, try2

# with open('./src/examples/latest (18).jpg', 'rb') as f:
with open('./src/examples/capture.jpg', 'rb') as f:
    image_bytes = f.read()

# undistorted = undistort_image(image_bytes)

# annotated_image_bytes, ids, corners = detect_aruco(undistorted)
# print("Detected ArUco IDs:", ids)
# with open('./src/examples/annotated_image.jpg', 'wb') as f:
#     f.write(annotated_image_bytes)

MARKER_SIZE=0.036  # in meters
MARKER_SPACING=0.005

grid = np.arange(0, 20).reshape((5,4))
marker_grid = [[[(MARKER_SIZE+MARKER_SPACING)*x, (MARKER_SIZE+MARKER_SPACING)*y, 0] for x in range(0,4)] for y in range(0,5)]
marker_positions = {grid[y][x]: marker_grid[y][x] for y in range(grid.shape[0]) for x in range(grid.shape[1])}

#print grid
for y in range(grid.shape[0]):
    for x in range(grid.shape[1]):
        print(f"ID: {grid[y][x]} Position: {np.round(marker_grid[y][x], 3)}")

img = decoede_image(image_bytes)
# undistorted = undistort_image(img)
# cv.imwrite('./src/examples/undistorted.jpg', undistorted)
res_img, rvec, tvec = try2(img, MARKER_SIZE, MARKER_SPACING)

R, _ = cv.Rodrigues(rvec)

# Camera position in board frame
camera_position = -R.T @ tvec
print("Camera position relative to board:", camera_position.flatten())
