import cv2
import cv2.aruco as aruco
from matplotlib.pyplot import grid
import numpy as np
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_MATRIX_DIR = os.path.join(BASE_DIR, "camera_matrix.npy")
DIST_COEFFS_DIR = os.path.join(BASE_DIR, "dist_coeffs.npy")

def get_marker_positions(marker_size, marker_spacing, rows=5, cols=4):
    grid = np.arange(0, 20).reshape((rows,cols))
    marker_grid = [[[(marker_size+marker_spacing)*x, (marker_size+marker_spacing)*y, 0] for x in range(0,cols)] for y in range(0,rows)]
    marker_positions = {grid[y][x]: marker_grid[y][x] for y in range(grid.shape[0]) for x in range(grid.shape[1])}
    return marker_positions

def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def undistort_image(image):
    cameraMatrix = np.load(CAMERA_MATRIX_DIR)
    distCoeffs   = np.load(DIST_COEFFS_DIR)
    
    undistorted = cv2.undistort(image, cameraMatrix, distCoeffs)
    img = cv2.rotate(undistorted, cv2.ROTATE_90_CLOCKWISE)
    return img

def get_camera_position(img, marker_positions, marker_size):
    img_copy = img.copy()
    camera_matrix = np.load(CAMERA_MATRIX_DIR)
    dist_coeffs = np.load(DIST_COEFFS_DIR)

    marker_corners, img_points = get_all_markers(img, marker_positions, marker_size)
    
    if marker_corners is None or img_points is None:
        print("No markers detected or matched.")
        return img_copy, None

    for i, point in enumerate(img_points):
        if i%2==0:
            cv2.circle(img_copy, tuple(point.astype(int)), 5, (255,0,0), -1)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_corners,
        imagePoints=img_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    proj, _ = cv2.projectPoints(marker_corners, rvec, tvec, camera_matrix, dist_coeffs)
    for p in proj.reshape(-1, 2):
        cv2.circle(img_copy, tuple(p.astype(int)), 3, (0, 0, 255), -1)

    cv2.drawFrameAxes(img_copy, camera_matrix, dist_coeffs, rvec, tvec, 0.25)

    error = np.mean(np.linalg.norm(img_points - proj.reshape(-1, 2), axis=1))
    print("Reprojection error:", error)

    R, _ = cv2.Rodrigues(rvec)
    camera_position = -R.T @ tvec
    camera_position = camera_position.flatten()

    #swapping boards x and y
    camera_position = [camera_position[1], camera_position[0], -camera_position[2]]

    return img_copy, camera_position

def get_all_markers(img, marker_positions, marker_size=0.036):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())

    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(ids) == 0:
        return None, None

    marker_corners_3d = np.array([
        [0, 0, 0],                  # top-left
        [marker_size, 0, 0],        # top-right
        [marker_size, marker_size, 0],  # bottom-right
        [0, marker_size, 0]         # bottom-left
    ], dtype=np.float32)

    marker_corners = []
    image_points = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in marker_positions:
            continue

        origin = np.array(marker_positions[marker_id], dtype=np.float32)
        obj_points = marker_corners_3d + origin

        marker_corners.append(obj_points)
        image_points.append(corners[i][0].astype(np.float32))

    if len(marker_corners) == 0:
        return None, None

    marker_corners = np.vstack(marker_corners)
    image_points = np.vstack(image_points)

    return marker_corners, image_points