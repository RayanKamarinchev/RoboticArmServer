import math
import numpy as np
from scipy.optimize import minimize

# Robot arm segment lengths in cm
a = 0.117
b = 0.122
c = 0.127
e = 0.067
camera_offset = 0.016
baseElevation = 0.13
# Initial joint angles in degrees
delta = np.radians(76)


def get_initial_angles():
    alpha = np.radians(100-7)
    beta = np.radians(110)
    gamma = np.radians(80)
    theta = np.radians(0)
    psi = np.radians(0)
    return [alpha, beta, gamma, theta, psi]

def get_arm_vectors(alpha, beta, gamma, psi): 
    l2_angle = alpha + beta - np.pi
    #arm1
    l1 = a * np.array([np.cos(alpha), 0, np.sin(beta)])
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

def rotate_vec(vec, theta):
    vec_already_rotated = np.arctan2(vec[1], vec[0])
    full_rotation_angle = vec_already_rotated + theta
    radius = vec[0]/np.cos(vec_already_rotated)
    return np.array([radius*np.cos(full_rotation_angle), radius*np.sin(full_rotation_angle), vec[2]])

def move_to_position(initial_gripper_position_in_space, initial_angles, desired_coords):
    x, y, z = desired_coords
    initial_gripper_position_from_arm, initial_cam_rotation = get_gripper_coords_and_cam_rotation_from_arm(initial_angles)
    print(f"Moving to position x:{x} y:{y} z:{z}")
    
    def objective(vars):
        position_from_arm, camera_angles = get_gripper_coords_and_cam_rotation_from_arm(vars)
        position_in_space = np.array([initial_gripper_position_from_arm[0] + initial_gripper_position_in_space[0] - position_from_arm[0], position_from_arm[1], position_from_arm[2]])
        position_diff = np.linalg.norm(position_in_space-np.array([x,y,z]))
        #TODO camera difference
        penalty = np.abs(vars[3]) + np.abs(vars[4])*3

        return position_diff + 1e-4 * penalty

    bounds = [
        (np.radians(-7), np.radians(-7+180)),       # alpha
        (np.radians(35), np.radians(35+180)),       # beta
        (np.radians(74), np.radians(74+180)),       # gamma
        (np.radians(-150), np.radians(-150+180)),   # theta
        (np.radians(-30), np.radians(-30+180))      # psi
    ]

    # Solve
    result = minimize(
        objective,
        initial_angles,
        bounds=bounds,
    )

    if result.success:
        alpha, beta, gamma, theta, psi  = result.x
        print("Solution found:")
        print(f"alpha = {np.degrees(alpha):.2f}°")
        print(f"beta = {np.degrees(beta):.2f}°")
        print(f"gamma = {np.degrees(gamma):.2f}°")
        print(f"theta = {np.degrees(theta):.2f}°")
        print(f"psi = {np.degrees(psi):.2f}°")
        
        pos, cam_rotation = get_gripper_coords_and_cam_rotation_from_arm(result.x)
        print(f"Initial gripper position from arm", initial_gripper_position_from_arm)
        print(f"Initial gripper position in space", initial_gripper_position_in_space)
        print(pos[0])
        pos[0] = initial_gripper_position_from_arm[0] + initial_gripper_position_in_space[0] - pos[0]
        print(f"Desired x: {x}")
        print(f"Result x: {pos[0]}")
        
        print(f"Desired y: {y}")
        print(f"Result y: {pos[1]}")
        
        print(f"Desired z: {z}")
        print(f"Result z: {pos[2]}")
    else:
        print("Optimization failed:", result.message)

    return [theta, alpha, beta, psi, gamma]

def get_move_angles(camera_coords, target_coords, current_angles):
    gripper_coords_in_space = conv_camera_coords_to_gripper_coords(camera_coords, current_angles)
    print(gripper_coords_in_space, "gripper coords in space")
    angles = move_to_position(gripper_coords_in_space, current_angles, target_coords)
    return angles

def conv_camera_coords_to_gripper_coords(camera_coords, angles):
    #TODO
    gripper_angle = angles[0] + angles[1] + angles[2]
    camera_angle = gripper_angle + delta
    
    _, arm_head = get_arm_vectors(angles[0], angles[1], angles[2], angles[4])
    _, camera_vector_direction = get_arm_vectors(angles[0], angles[1], angles[2] + delta, angles[4])
    camera_vector_normalized = camera_vector_direction * e / c
    caemra_offset = np.cross(camera_vector_normalized, arm_head)
    caemra_offset_normalized = caemra_offset / np.linalg.norm(caemra_offset) * camera_offset
    
    gripper_position = camera_coords-caemra_offset_normalized-camera_vector_normalized+arm_head
    
    return gripper_position
