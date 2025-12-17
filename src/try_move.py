from movement import get_move_angles, get_initial_angles, get_gripper_coords_and_cam_rotation_from_arm, conv_camera_coords_to_gripper_coords
import numpy as np

# camera_coords = [0.1343155369863135, 0.04369305141274703, 0.2867105330044601]
camera_coords = [0.14079992471301941, 0.06420141472130722, 0.28964597026748984]
# gripper_coors = conv_camera_coords_to_gripper_coords(camera_coords, get_initial_angles())
# desired_coords = gripper_coors
desired_coords = [-0.007098067481364739, 0.07774760600851423, 0.3377893786785457]
angles = get_move_angles(camera_coords, desired_coords, get_initial_angles())

theta, alpha, beta, psi, gamma = angles

print("result")
print([alpha, beta, gamma, theta, psi])
print(get_gripper_coords_and_cam_rotation_from_arm(np.radians([alpha, beta, gamma, theta, psi]))[0])

print("initial")
alpha, beta, gamma, theta, psi = get_initial_angles()
print(np.degrees([alpha, beta, gamma, theta, psi]))
print(get_gripper_coords_and_cam_rotation_from_arm([alpha, beta, gamma, theta, psi])[0])
# print("result z")
# print(np.sin(alpha)*a+np.sin(alpha+beta-180)*b+np.sin(alpha+beta+gamma)*c)





#another test
theta, alpha, beta, psi, gamma = np.radians([12.0, 66.0, 166.0, -19.0, 139.0])
print("real")
print(get_gripper_coords_and_cam_rotation_from_arm([alpha, beta, gamma, theta, psi])[0])
