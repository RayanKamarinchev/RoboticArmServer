from movement import get_move_angles, get_initial_angles, get_gripper_coords_and_cam_rotation_from_arm, conv_camera_coords_to_gripper_coords
import numpy as np
a = 0.117
b = 0.122
c = 0.127
e = 0.067

# camera_coords = [0.1343155369863135, 0.04369305141274703, 0.2867105330044601]
camera_coords = [0.1343155369863135, 0.04369305141274703, 0.2867105330044601]
# desired_coords = [0.2, camera_coords[1], 0.05]
gripper_coors = conv_camera_coords_to_gripper_coords(camera_coords, get_initial_angles())
desired_coords = gripper_coors
angles = get_move_angles(camera_coords, desired_coords, get_initial_angles())
theta, alpha, beta, psi, gamma = angles

print("result")
print(np.degrees([alpha, beta, gamma, theta, psi]))
print(get_gripper_coords_and_cam_rotation_from_arm([alpha, beta, gamma, theta, psi])[0])

print("result z")
print(np.sin(alpha)*a+np.sin(alpha+beta-180)*b+np.sin(alpha+beta+gamma)*c)


print("initial")
alpha, beta, gamma, theta, psi = get_initial_angles()
print(np.degrees([alpha, beta, gamma, theta, psi]))
print(get_gripper_coords_and_cam_rotation_from_arm([alpha, beta, gamma, theta, psi])[0])
print("result z")
print(np.sin(alpha)*a+np.sin(alpha+beta-180)*b+np.sin(alpha+beta+gamma)*c)
