import cv2 as cv
import numpy as np
from camera_utils import get_camera_position, undistort_image, decode_image, get_marker_positions

# with open('./src/examples/empty.jpg', 'rb') as f:
with open('./src/examples/latest (3).jpg', 'rb') as f:
    image_bytes = f.read()

# undistorted = undistort_image(image_bytes)

# annotated_image_bytes, ids, corners = detect_aruco(undistorted)
# print("Detected ArUco IDs:", ids)
# with open('./src/examples/annotated_image.jpg', 'wb') as f:
#     f.write(annotated_image_bytes)

MARKER_SIZE=0.036  # in meters
MARKER_SPACING=0.005

#print grid
# for y in range(grid.shape[0]):
#     for x in range(grid.shape[1]):
#         print(f"ID: {grid[y][x]} Position: {np.round(marker_grid[y][x], 3)}")
marker_positions = get_marker_positions(MARKER_SIZE, MARKER_SPACING)

img = decode_image(image_bytes)
# undistorted = undistort_image(img)
res_img1, camera_position = get_camera_position(img, marker_positions, MARKER_SIZE)
# res_img2, camera_position = get_camera_pos_from_board(img, MARKER_SIZE, MARKER_SPACING)

cv.imwrite('./src/examples/annotated_image.jpg', res_img1)
# cv.imwrite('./src/examples/annotated_image_2.jpg', res_img2)


print("Camera position relative to board:", camera_position)


#test marker count
all_marker_positions = marker_positions.copy()

_, ground_truth = get_camera_position(img, all_marker_positions, MARKER_SIZE)
for i in range(0, 19):
    marker_positions.pop(i)
    _, estimated = get_camera_position(img, marker_positions, MARKER_SIZE)
    print(f"Removed marker {i}, difference: {np.linalg.norm(np.array(ground_truth)-np.array(estimated))} meters")
