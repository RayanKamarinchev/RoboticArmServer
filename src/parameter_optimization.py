import os
import cv2
import numpy as np
from scipy.optimize import minimize

from camera_utils import get_camera_position, get_marker_positions
MARKER_SIZE = 0.036
MARKER_SPACING = 0.005
baseElevation = 0.132
e = 0.07#68-70
camera_offset_len = 0.016
delta = np.radians(75)

def conv_camera_coords_to_gripper_coords(camera_coords, angles, coordinate_systems_angle, a, b, c):
    gripper_angle = angles[0] + angles[1] + angles[2]
    
    _, arm_head = get_arm_vectors(angles[0], angles[1], angles[2], angles[4], a, b, c)
    _, camera_vector_direction = get_arm_vectors(angles[0], angles[1], angles[2] + delta, angles[4], a, b, c)
    camera_vector_normalized = camera_vector_direction * e / c
    camera_offset = np.cross(arm_head, camera_vector_normalized)
    caemra_offset_normalized = camera_offset / np.linalg.norm(camera_offset) * camera_offset_len
    disposition_vec_in_arm_system = -caemra_offset_normalized-camera_vector_normalized+arm_head
    
    co, si = np.cos(coordinate_systems_angle), np.sin(coordinate_systems_angle)
    mat = np.array([
        [-co, si, 0],
        [si,  co, 0],
        [0,  0, 1]
    ])
    disposition_vec = mat @ disposition_vec_in_arm_system
    gripper_position = camera_coords + disposition_vec
    return gripper_position



def rotate_vec(vec, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R @ vec

def get_rotation_matrix(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [-c, s, 0],
        [ s, c, 0],
        [0, 0, 1]
    ])
    
def get_translation(p1, p2, alpha):
    R = get_rotation_matrix(alpha)
    return p2 - R @ p1

def transform_arm_to_space_coords(p1, alpha, t):
    R = get_rotation_matrix(alpha)
    return R @ p1 + t

def transform_space_to_arm_coords(p2, alpha, t):
    R = get_rotation_matrix(alpha)
    return R.T @ (p2 - t)

def get_arm_vectors(alpha, beta, gamma, psi, a, b, c): 
    l2_angle = alpha + beta - np.pi
    #arm1
    l1 = a * np.array([np.cos(alpha), 0, np.sin(alpha)])
    #arm2
    l2 = b * np.array([np.cos(l2_angle),0, np.sin(l2_angle)])
    #combined base arm
    lb = l1 + l2
    #angle above the xy plane (first z then x)
    phi = np.arctan2(lb[2], lb[0])
    #head arm angle relative to x
    angle_sum = alpha + beta + gamma
    #head elevation from base arm
    epsilon = np.pi/2 + angle_sum - phi
    #head arm components
    lh_x = c * np.array([np.cos(angle_sum), 0, np.sin(angle_sum)])
    lh_y = c * np.array([np.sin(epsilon) * np.cos(phi), np.cos(epsilon), np.sin(epsilon) * np.sin(phi)])
    lh = np.cos(psi) * lh_x + np.sin(psi) * lh_y
    
    return lb, lh

def get_gripper_coords_and_cam_rotation_from_arm(angles, a, b, c):
    alpha, beta, gamma, theta, psi = angles
    psi=0
    
    lb, lh = get_arm_vectors(alpha, beta, gamma, psi, a, b, c)
    
    arm_vector = lb + lh
    #calculating the rotation angle around the z axis
    position = rotate_vec(arm_vector, theta)
    #calculate camera angles and rotation
    base_arm_rotated = rotate_vec(lb, theta)
    head_rotated = position-base_arm_rotated
    azimuth = np.arctan2(head_rotated[1], head_rotated[0])
    radius = np.sqrt(head_rotated[0]**2 + head_rotated[1]**2)
    elevation = np.arctan2(head_rotated[2], radius)
    rotation = psi
    
    position[2] += baseElevation
    return (position, [azimuth, elevation, rotation])

def get_angles(vars, servo_angles):
    return [np.radians(servo_angles[0]+vars[0]),np.radians( vars[1]-servo_angles[1]), np.radians(servo_angles[2]+vars[2]), np.radians(servo_angles[3]-30), 0]

cam_positions = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder =  os.path.join(BASE_DIR, "views_from_servo_angles")

images = [os.path.join(folder, f) for f in os.listdir(folder)]
initial_img_path = next((x for x in images if '30_100_100_6' in x))

initial_img = cv2.imread(initial_img_path)
_, initial_camera_position, initial_coordinate_systems_angle, _, _, _ = get_camera_position(initial_img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
print("Angle: ", initial_coordinate_systems_angle)
print("Init camera position: ", initial_camera_position)

#alpha offset, beta offset, gamma offset, a, b, c
init_vars = [-8, 211, 73, 0.119, 0.122, 0.13]

imgs_info = []

for img_path in images:
    img = cv2.imread(img_path)
    file_name = os.path.splitext(os.path.basename(img_path))[0]
    angles_unordered = [int(x) for x in file_name.split('_')]
    servo_angles = angles_unordered[1:] + angles_unordered[:1]
    
    _, cam_position, coordinate_systems_angle, _, _, _ = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    imgs_info.append([servo_angles, cam_position, coordinate_systems_angle])

def objective(vars, initial_camera_position, initial_coordinate_systems_angle, imgs_info):
    
    initial_angles = get_angles(vars, [100, 100, 6, 30])
    initial_gripper_position_in_space = conv_camera_coords_to_gripper_coords(initial_camera_position, initial_angles, initial_coordinate_systems_angle, vars[3], vars[4], vars[5])
    # print("Initial grip position in space", initial_gripper_position_in_space)

    initial_gripper_position_from_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(initial_angles, vars[3], vars[4], vars[5])
    # print("Initial angles: ", initial_angles)
    # print("Initial grip position from arm", initial_gripper_position_from_arm)

    arm_angle = np.arctan2(initial_gripper_position_from_arm[1], initial_gripper_position_from_arm[0])
    # print("Arm angle", np.degrees(arm_angle))

    # print("arm vec in space", get_rotation_matrix(initial_coordinate_systems_angle) @ initial_gripper_position_from_arm)
    translation = get_translation(initial_gripper_position_from_arm, initial_gripper_position_in_space, initial_coordinate_systems_angle - arm_angle)
    # print("Translation: ", translation)
    cumulative_error = 0

    for info in imgs_info:
        
        servo_angles, cam_position, coordinate_systems_angle = info
        
        angles = get_angles(vars, servo_angles)
        real_gripper_position_in_space = conv_camera_coords_to_gripper_coords(cam_position, angles, coordinate_systems_angle, vars[3], vars[4], vars[5])
        
        position_from_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(angles, vars[3], vars[4], vars[5])
        #convert from arm coordinate system to from board(in space) coordinate system
        position_in_space = transform_arm_to_space_coords(position_from_arm, coordinate_systems_angle, translation)
        
        position_diff = np.linalg.norm(position_in_space-real_gripper_position_in_space)
        # print("Camera position: ", cam_position)
        # print("Real gripper position: ", real_gripper_position_in_space)
        # print("Calculated gripper position: ", position_in_space)
        # print("Error: ", position_diff)
        # position_diff*=10e4
        cumulative_error+=position_diff
        
    cumulative_error/=len(images)
    # print(cumulative_error)
    return cumulative_error
        
        
print("----------------------------------------------------------")
result = minimize(
        objective,
        init_vars,
        args=(initial_camera_position, initial_coordinate_systems_angle, imgs_info)
    )

print(result)


err = objective(result.x, initial_camera_position, initial_coordinate_systems_angle, imgs_info)
print("err: ", err)
err2 = objective(init_vars, initial_camera_position, initial_coordinate_systems_angle, imgs_info)
print("err2: ", err2)