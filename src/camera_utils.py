import cv2
import cv2.aruco as aruco
import numpy as np
import io
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def undistort_image(image):
    cameraMatrix = np.load(os.path.join(BASE_DIR, "camera_matrix.npy"))
    distCoeffs   = np.load(os.path.join(BASE_DIR, "dist_coeffs.npy"))
    
    undistorted = cv2.undistort(image, cameraMatrix, distCoeffs)
    img = cv2.rotate(undistorted, cv2.ROTATE_90_CLOCKWISE)
    return img

def detect_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        print("Detected ArUco IDs:", ids.flatten())
        aruco.drawDetectedMarkers(img, corners, ids)
    else:
        print("No ArUco markers detected.")

    cv2.imwrite('./src/examples/annotated_image1.jpg', img)
    return img, ids, corners if ids is not None else []

def get_camera_position(img, marker_positions=None, marker_size=0.036, marker_id=16):
    camera_matrix = np.load(os.path.join(BASE_DIR, "camera_matrix.npy"))
    dist_coeffs   = np.load(os.path.join(BASE_DIR, "dist_coeffs.npy"))
    
    marker_corners, image_points = get_all_markers(img, marker_size, marker_spacing)
    
    for i, point in enumerate(image_points):
        if i%2==0:
            cv2.circle(img, tuple(point.astype(int)), 5, (255,0,0), -1)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_corners,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,   # <-- must use real distortion
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    proj, _ = cv2.projectPoints(marker_corners, rvec, tvec, camera_matrix, dist_coeffs)
    
    for p in proj.reshape(-1,2):
        cv2.circle(img, tuple(p.astype(int)), 3, (0,0,255), -1)
        
    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.25)
    
    error = np.mean(np.linalg.norm(image_points - proj.reshape(-1,2), axis=1))

    print("Position error: ", error)
    
    R, _ = cv2.Rodrigues(rvec)
    # Camera position in board frame
    camera_position = -R.T @ tvec
    camera_position = camera_position.flatten()
    #swapping boards x and y
    camera_position = [camera_position[1], camera_position[0], -camera_position[2]]
    return img, camera_position

def get_all_markers(corners, ids, marker_positions, marker_size=0.036):
    if ids is None or len(ids) == 0:
        return None, None
    
    marker_corners_3d = np.array([
        [0, 0, 0],
        [ marker_size,  0, 0],
        [ marker_size, -marker_size, 0],
        [0, -marker_size, 0]
    ], dtype=np.float32)

    all_obj = []
    all_img = []

    for i, marker_id in enumerate(ids.flatten()):
        if marker_id not in marker_positions:
            continue

        marker_origin = np.array(marker_positions[marker_id], dtype=np.float32)
        obj_points = marker_corners_3d + marker_origin

        all_obj.extend(obj_points)
        all_img.extend(corners[i][0].astype(np.float32)) 
    
    all_obj = np.array(all_obj, dtype=np.float32)
    all_img = np.array(all_img, dtype=np.float32)

    return all_obj, all_img

def aruco_board(image, marker_size=0.036, marker_spacing=0.005):
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    board = aruco.GridBoard((4,5), markerLength=marker_size, markerSeparation=marker_spacing, dictionary=dictionary)
    print("Board size:", board.getGridSize())

    # Detect markers
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, rejected = detector.detectMarkers(image)
    # Match the detected markers with the board model
    marker_corners, image_points = board.matchImagePoints(corners, ids)
    marker_corners = marker_corners.reshape(-1, 3)
    image_points = image_points.reshape(-1, 2)
    return marker_corners, image_points






def get_camera_pos_from_board(img, marker_size=0.036, marker_spacing=0.005):
    camera_matrix = np.load(os.path.join(BASE_DIR, "camera_matrix.npy"))
    dist_coeffs   = np.load(os.path.join(BASE_DIR, "dist_coeffs.npy"))
    
    marker_corners, image_points = aruco_board(img, marker_size, marker_spacing)
    
    for i, point in enumerate(image_points):
        if i%2==0:
            cv2.circle(img, tuple(point.astype(int)), 5, (255,0,0), -1)

    success, rvec, tvec = cv2.solvePnP(
        objectPoints=marker_corners,
        imagePoints=image_points,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,   # <-- must use real distortion
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    proj, _ = cv2.projectPoints(marker_corners, rvec, tvec, camera_matrix, dist_coeffs)
    
    for p in proj.reshape(-1,2):
        cv2.circle(img, tuple(p.astype(int)), 3, (0,0,255), -1)
        
    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.25)
    
    error = np.mean(np.linalg.norm(image_points - proj.reshape(-1,2), axis=1))

    print("Position error: ", error)
    
    R, _ = cv2.Rodrigues(rvec)
    # Camera position in board frame
    camera_position = -R.T @ tvec
    camera_position = camera_position.flatten()
    #swapping boards x and y
    camera_position = [camera_position[1], camera_position[0], -camera_position[2]]
    return img, camera_position