import cv2 as cv
import numpy as np
from camera_utils import get_camera_position, undistort_image, decode_image, get_marker_positions
from movement import conv_camera_coords_to_gripper_coords, get_arm_vectors, get_gripper_coords_and_cam_rotation_from_arm, get_initial_angles, get_rotation_matrix, get_translation, transform_arm_to_space_coords
from scipy.optimize import minimize

c = 0.13
e = 0.07#68-70
delta = np.radians(75) 
camera_offset_len = 0.016

def angle_between(v1, v2):
    #using the cosine theorem
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

# with open('./src/examples/empty.jpg', 'rb') as f:
# with open('./src/examples/latest (3).jpg', 'rb') as f:
# with open('./uploads/15_image_30.00_100.00_100.00_155.00_6.00_160.00.jpg', 'rb') as f:
# with open('./uploads/latest.jpg', 'rb') as f:
with open('./uploads/try.jpg', 'rb') as f:
    image_bytes = f.read()

# undistorted = undistort_image(image_bytes)

# annotated_image_bytes, ids, corners = detect_aruco(undistorted)
# print("Detected ArUco IDs:", ids)
# with open('./src/examples/annotated_image.jpg', 'wb') as f:
#     f.write(annotated_image_bytes)

MARKER_SIZE=0.036  # in meters
MARKER_SPACING=0.005

# coordinate_systems_angle = 0.38041146105718765
# co, si = np.cos(coordinate_systems_angle), np.sin(coordinate_systems_angle)
# cam = np.array([0.11160683, 0.05119527, 0.30186553])
# dis = np.array([-0.05138573,  0.016,      -0.12020399])
# mat = np.array([
#         [ -co, si, 0],
#         [ si,  co, 0],
#         [0,  0, 1]
#     ])
# print(mat @ dis)#[ 0.04730703  0.02566199 -0.12020399]


img = decode_image(image_bytes)
# undistorted = undistort_image(img)

res_img1, camera_position, coordinate_systems_angle, _, _, _ = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
print("Camera position:", camera_position)
cv.imwrite("try.png", res_img1)
initial_gripper_position_in_space = conv_camera_coords_to_gripper_coords(camera_position, get_initial_angles(), coordinate_systems_angle)

print("Gripper position in space", initial_gripper_position_in_space)
initial_gripper_position_from_arm, initial_cam_rotation = get_gripper_coords_and_cam_rotation_from_arm(get_initial_angles())

# coordinate_systems_angle = np.radians(coordinate_systems_angle)
# print("Angle: ", coordinate_systems_angle)

# arm_angle = np.arctan2(initial_gripper_position_from_arm[1], initial_gripper_position_from_arm[0])
# print("Arm angle", np.degrees(arm_angle))

# coordinate_systems_angle -= arm_angle

print("Initial grip position from arm", initial_gripper_position_from_arm)
# print("Initial grip position in space", initial_gripper_position_in_space)
# print("arm vec in space", get_rotation_matrix(coordinate_systems_angle) @ initial_gripper_position_from_arm)
# translation = get_translation(initial_gripper_position_from_arm, initial_gripper_position_in_space, coordinate_systems_angle)
# print("Translation: ", translation)


cv.imwrite('./src/examples/annotated_image.jpg', res_img1)

# print("Angle", angle_from_cam)

# azimuth = np.arctan2(camera_position[0], camera_position[1])
# azimuth = (azimuth + 2*np.pi) % (2*np.pi)
# print(np.degrees(azimuth), "az2")
    
# forward = camera_rotation[:, 2]

# azimuth = np.arctan2(forward[1], forward[0])
# azimuth = (azimuth + 2*np.pi) % (2*np.pi)
# print(np.degrees(azimuth), "az3")

# azimuth = np.arctan2(forward[0], forward[1])
# azimuth = (azimuth + 2*np.pi) % (2*np.pi)
# print(np.degrees(azimuth), "az4")

# forward = camera_rotation @ np.array([0, 0, 1])  # camera optical axis

# # project onto ground plane
# forward[2] = 0
# forward /= np.linalg.norm(forward)

# azimuth = np.arctan2(forward[1], forward[0])
# print(np.degrees(azimuth), "az5")
# azimuth = np.arctan2(forward[0], forward[1])
# print(np.degrees(azimuth), "az6")


#test marker count
# all_marker_positions = marker_positions.copy()

# _, ground_truth, _ = get_camera_position(img, all_marker_positions, MARKER_SIZE)
# for i in range(0, 19):
#     marker_positions.pop(i)
#     _, estimated, _ = get_camera_position(img, marker_positions, MARKER_SIZE)
#     print(f"Removed marker {i}, difference: {np.linalg.norm(np.array(ground_truth)-np.array(estimated))} meters")



#initial angles finding
target = np.asarray(initial_gripper_position_in_space)

def objective(angles):
    pos_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(angles)

    error = pos_arm[2] - target[2]
    return np.dot(error, error)  # squared error (smooth!)

bounds = [
    (np.radians(-10),  np.radians(170)),   # alpha
    (np.radians(34),   np.radians(214)),   # beta
    (np.radians(73),   np.radians(253)),   # gamma
    (np.radians(0), np.radians(0)),    # theta
    (np.radians(0),  np.radians(0)),   # psi
]

result = minimize(
    objective,
    get_initial_angles(),
    method="L-BFGS-B",
    bounds=bounds
)

if not result.success and result.fun > 1e-10:
    raise RuntimeError(
        f"Calibration failed: {result.message}, error={result.fun}"
    )
    
print("Initial angles (degrees):", np.degrees(get_initial_angles()))
print("Calibrated angles (degrees):", np.degrees(result.x))