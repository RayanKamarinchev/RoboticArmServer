from movement import get_move_angles, get_initial_angles, get_gripper_coords_and_cam_rotation_from_arm, conv_camera_coords_to_gripper_coords
import numpy as np

# camera_coords = [0.1343155369863135, 0.04369305141274703, 0.2867105330044601]
camera_coords = [0.13837233502768634, 0.047776567903755555, 0.2875156334661047]
gripper_coors = conv_camera_coords_to_gripper_coords(camera_coords, get_initial_angles())
target_coords = gripper_coors
target_coords[1] -= 0.01

angles = get_move_angles(camera_coords, target_coords, get_initial_angles())
theta, alpha, beta, psi, gamma = angles

print("result")
print([alpha, beta, gamma, theta, psi])
print(get_gripper_coords_and_cam_rotation_from_arm(np.radians([alpha, beta, gamma, theta, psi]))[0])

print("initial")
alpha, beta, gamma, theta, psi = get_initial_angles()
print(np.degrees([alpha, beta, gamma, theta, psi]))
print(get_gripper_coords_and_cam_rotation_from_arm([alpha, beta, gamma, theta, psi])[0])