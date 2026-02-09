from movement import conv_camera_coords_to_gripper_coords
import numpy as np

alpha = np.radians(100)#90
beta = np.radians(111)#109
gamma = np.radians(79)#79
theta = np.radians(0)
psi = np.radians(0)
angles = [alpha, beta, gamma, theta, psi]

cam_coords = [0.13430879, 0.04383473, 0.25820867]
grip_coords = conv_camera_coords_to_gripper_coords(cam_coords, angles, 0)
print(grip_coords)