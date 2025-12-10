import cv2
import cv2.aruco as aruco
import numpy as np
import io

def decode_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def undistort_image(image):
    cameraMatrix = np.load("./src/camera_matrix.npy")
    distCoeffs   = np.load("./src/dist_coeffs.npy")
    
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
    img = undistort_image(img)
    
    # img, ids, corners = detect_aruco(img)
    # if ids is None or len(ids) == 0:
    #     print("No markers detected for pose estimation.")
    #     return
    camera_matrix = np.load("./src/camera_matrix.npy")
    dist_coeffs   = np.load("./src/dist_coeffs.npy")
    
    marker_corners, image_points = aruco_board(img, marker_size, marker_spacing=0.005)
    
    
    # dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
    # corners, ids, _ = aruco.ArucoDetector(dictionary, aruco.DetectorParameters()).detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # marker_corners, image_points = get_all_markers(corners, ids, marker_positions, marker_size)
    
    # undistort points
    # image_points = cv2.undistortPoints(
    #     np.expand_dims(image_points, axis=1),
    #     cameraMatrix,
    #     distCoeffs,
    #     P=cameraMatrix
    # ).reshape(-1, 2)
    
    #draw them
    # for point in image_points:
    #     cv2.circle(img, tuple(point.astype(int)), 5, (0,255,0), -1)
        
    # #show all obj points
    # for point in marker_corners:
    #     proj_point, _ = cv2.projectPoints(np.array([point]), np.zeros((3,1)), np.zeros((3,1)), np.load("./src/camera_matrix.npy"), np.load("./src/dist_coeffs.npy"))
    #     cv2.circle(img, tuple(proj_point[0][0].astype(int)), 3, (255,0,0), -1)
        
    for i, point in enumerate(image_points):
        if i%2==0:
            cv2.putText(img, str(i), tuple(point.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
            # cv2.circle(img, tuple(point.astype(int)), 4, (0,255,0), -1)
        
            proj_point, _ = cv2.projectPoints(np.array([marker_corners[i]]), np.zeros((3,1)), np.zeros((3,1)), np.load("./src/camera_matrix.npy"), None)
            cv2.putText(img, str(i), tuple(proj_point[0][0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    
    
    success, rvec, tvec = cv2.solvePnP(
        marker_corners,      # 3D points
        image_points,        # 2D points
        camera_matrix,        # intrinsic matrix
        None,           # distortion coefficients
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    proj_points, _ = cv2.projectPoints(marker_corners, rvec, tvec, camera_matrix, None)
    for p in proj_points.reshape(-1,2):
        cv2.circle(img, tuple(p.astype(int)), 3, (0,0,255), -1)
    cv2.imwrite('./src/examples/annotated_image.jpg', img)

    proj_points = proj_points.reshape(-1, 2)
    error = np.linalg.norm(image_points - proj_points, axis=1).mean()
    print("Reprojection error (pixels):", error)

    if success:
        print("Translation vector:", tvec)
    else:
        print("Pose estimation failed")
        
    return img

def get_single_marker(corners, ids, marker_id, marker_size=0.036):
    if ids is None or len(ids) == 0:
        return None, None
    
    marker_corners_3d = np.array([
        [-marker_size/2,  marker_size/2, 0],
        [ marker_size/2,  marker_size/2, 0],
        [ marker_size/2, -marker_size/2, 0],
        [-marker_size/2, -marker_size/2, 0]
    ], dtype=np.float32)
    
    marker_idx = ids.flatten().tolist().index(marker_id)
    image_points = corners[marker_idx].reshape(-1, 2).astype(np.float32)
    
    return marker_corners_3d, image_points

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
    camera_matrix = np.load("./src/camera_matrix.npy")
    dist_coeffs   = np.load("./src/dist_coeffs.npy")
    
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