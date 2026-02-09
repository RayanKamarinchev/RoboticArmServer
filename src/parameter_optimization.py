import os
import cv2
import numpy as np
from scipy.optimize import minimize

from camera_utils import get_camera_position, get_marker_positions
from movement import conv_camera_coords_to_gripper_coords
MARKER_SIZE = 0.036
MARKER_SPACING = 0.005
baseElevation = 0.132


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

def get_gripper_coords_and_cam_rotation_from_arm(angles):
    alpha, beta, gamma, theta, psi = angles
    psi=0
    
    lb, lh = get_arm_vectors(alpha, beta, gamma, psi)
    
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


def move_to_position(initial_gripper_position_in_space, initial_angles, desired_coords, coordinate_systems_angle):
    x, y, z = desired_coords
    initial_gripper_position_from_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(initial_angles)
    print(f"Moving to position x:{x} y:{y} z:{z}")
    print("Angle: ", coordinate_systems_angle)
    
    arm_angle = np.arctan2(initial_gripper_position_from_arm[1], initial_gripper_position_from_arm[0])
    print("Arm angle", np.degrees(arm_angle))
    
    coordinate_systems_angle -= arm_angle
    print("Initial grip position from arm", initial_gripper_position_from_arm)
    print("Initial grip position in space", initial_gripper_position_in_space)
    print("arm vec in space", get_rotation_matrix(coordinate_systems_angle) @ initial_gripper_position_from_arm)
    translation = get_translation(initial_gripper_position_from_arm, initial_gripper_position_in_space, coordinate_systems_angle)
    print("Translation: ", translation)
    def objective(vars):
        position_from_arm, camera_angles = get_gripper_coords_and_cam_rotation_from_arm(vars)
        #convert from arm coordinate system to from board(in space) coordinate system
        position_in_space = transform_arm_to_space_coords(position_from_arm, coordinate_systems_angle, translation)
        
        position_diff = np.linalg.norm(position_in_space-np.array([x,y,z]))
        # position_diff = np.linalg.norm(position_in_space-np.array([x,-y,z]))
        #TODO camera difference
        # penalty = np.linalg.norm(vars - initial_angles)
        penalty = np.abs(vars[3]) + np.abs(vars[4])*3
        # penalty = 5*vars[0]-np.round(vars[0])
        return position_diff + 1e-4 * penalty

    bounds = [
        (np.radians(-10), np.radians(-10+180)),       # alpha
        (np.radians(34), np.radians(34+180)),       # beta
        (np.radians(73), np.radians(73+180)),       # gamma
        (np.radians(-150), np.radians(-150+180)),   # theta
        # (np.radians(-30), np.radians(-30+180))      # psi
        (np.radians(0), np.radians(0))      # psi
    ]

    result = minimize(
        objective,
        initial_angles,
        bounds=bounds,
    )

    if result.success or result.fun < 1e-6:
        print("angles:")
        print(np.round(np.degrees(result.x),  decimals=1))
        alpha, beta, gamma, theta, psi  = np.round(np.degrees(result.x))
        
        position_from_arm, camera_angles = get_gripper_coords_and_cam_rotation_from_arm(result.x)
        print("x grip in space", np.sin(coordinate_systems_angle)*initial_gripper_position_in_space[0])
        print("y grip in space", -np.cos(coordinate_systems_angle)*initial_gripper_position_in_space[1])
        print("y grip in arm", - initial_gripper_position_from_arm[1])
        print("pos", - position_from_arm[1])
        
        pos, cam_rotation = get_gripper_coords_and_cam_rotation_from_arm(result.x)
        print(f"Initial gripper position from arm", initial_gripper_position_from_arm)
        print(f"Initial gripper position in space", initial_gripper_position_in_space)
        pos[0] = initial_gripper_position_from_arm[0] + initial_gripper_position_in_space[0] - pos[0]
        pos[1] = initial_gripper_position_in_space[1] + pos[1]
    else:
        print("Optimization failed:", result.message)

    return [theta, alpha, beta, psi, gamma]

def get_move_angles(initial_camera_coords, target_coords, initial_angles, coordinate_systems_angle):
    gripper_coords_in_space = conv_camera_coords_to_gripper_coords(initial_camera_coords, initial_angles, coordinate_systems_angle)
    print("Gripper coords in space", gripper_coords_in_space)
    angles = move_to_position(gripper_coords_in_space, initial_angles, target_coords, coordinate_systems_angle)
    return angles


cam_positions = []
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder =  os.path.join(BASE_DIR, "views_from_servo_angles")

images = [os.path.join(folder, f) for f in os.listdir(folder)]

initial_img = cv2.imread("30_100_100_6.jpg")
_, initial_camera_position, cam_angle, _, _, _ = get_camera_position(initial_img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)

#alpha offset, beta offset, gamma offset, a, b, c
vars = [-8, 211, 73, 0.119, 0.122, 0.13]

def get_angles(vars, servo_angles):
    return [servo_angles[0]+vars[0], vars[1]-servo_angles[1], servo_angles[2]+vars[2], servo_angles[3]-30, 0]

initial_gripper_position = conv_camera_coords_to_gripper_coords(initial_camera_position, get_angles(vars, [100, 100, 6, 30]), cam_angle)

for img_path in images:
    img = cv2.imread(img_path)
    _, cam_position, _, _, _, _ = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    print(f"Image: {img_path}   Cam position: {cam_position}")