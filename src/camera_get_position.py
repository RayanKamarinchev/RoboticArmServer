import cv2
import cv2.aruco as aruco
import numpy as np
import io

def detect_aruco(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
    parameters = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(img, corners, ids)

        for corner, marker_id in zip(corners, ids.flatten()):
            c = corner[0]
            center = c.mean(axis=0).astype(int)
            cv2.circle(img, tuple(center), 5, (0, 0, 255), -1)
            cv2.putText(img, str(marker_id), tuple(center + np.array([10, 10])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        print(f"Detected ArUco IDs: {ids.flatten().tolist()}")
    else:
        print("No ArUco markers detected.")

    success, encoded_img = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError("Failed to encode annotated image")

    return encoded_img.tobytes(), ids

def get_camera_position(image_stream):
    return [0.0, 0.0, 0.0]