import math
import numpy as np
from scipy.optimize import minimize

# Robot arm segment lengths in cm
a = 0.117
b = 0.122
c = 0.127
e = 0.067
baseElevation = 0.13
# Initial joint angles in degrees
alpha = np.radians(100-7)
beta = np.radians(100)
gamma = np.radians(80)
delta = np.radians(76)

def get_initial_angles():
    return [alpha, beta, gamma]

def conv_camera_coords_to_gripper_coords(camera_coords, angles):
    gripper_angle = angles[0] + angles[1] + angles[2]
    camera_angle = gripper_angle + delta
    camera_vertical_change = math.sin(camera_angle)  * e
    camera_horizontal_change = math.cos(camera_angle) * e
    gripper__base_x = camera_coords[0] + camera_horizontal_change
    gripper__base_z = camera_coords[2] - camera_vertical_change
    
    gripper_horizontal_change = math.cos(gripper_angle) * c
    gripper_vertical_change = math.sin(gripper_angle) * c
    gripper_x = gripper__base_x - gripper_horizontal_change
    gripper_z = gripper__base_z + gripper_vertical_change
    
    return [gripper_x, camera_coords[1], gripper_z]

def x_from_arm(angles):
    return a * np.cos(angles[0]) - b * np.cos(angles[0] + angles[1]) + c * np.cos(angles[0] + angles[1] + angles[2])

def z_from_arm(angles):
    return a * np.sin(angles[0]) - b * np.sin(angles[0] + angles[1]) + c * np.sin(angles[0] + angles[1] + angles[2]) + baseElevation

def get_gripper_coords_from_arm(angles):
    z = z_from_arm(angles)
    x = x_from_arm(angles)
    return [x, 0, z]

def move_to_position(initial_gripper_coords_from_base, initial_angles, desired_coords):
    x, y, z = desired_coords
    print(f"Moving to position x:{x} y:{y} z:{z}")
    
    initial_arm_x, _, initial_arm_z = get_gripper_coords_from_arm(initial_angles)
    
    def objective(vars):
        alpha, beta, gamma = vars
        # Equation residuals
        eq1 = z_from_arm(vars) - z
        eq2 = initial_gripper_coords_from_base[0] - (x_from_arm(vars) - initial_arm_x) - x

        total_angle_deg = np.degrees(x + y + z)
        penalty = (total_angle_deg - 270) ** 2 # TODO review penalty

        return eq1 ** 2 + eq2 ** 2 + 1e-8 * penalty

    def constraintJoint1(vars):
        alpha, beta, gamma = vars
        return np.degrees(alpha) + 7

    def constraintJointUpper1(vars):
        alpha, beta, gamma = vars
        return 173 - np.degrees(alpha)

    def constraintJoint2(vars):
        alpha, beta, gamma = vars
        return np.degrees(beta) - 35

    def constraintJointUpper2(vars):
        alpha, beta, gamma = vars
        return 215 - np.degrees(beta)

    def constraintJoint3(vars):
        alpha, beta, gamma = vars
        return np.degrees(gamma) - 74

    def constraintJointUpper3(vars):
        alpha, beta, gamma = vars
        return 254 - np.degrees(gamma)

    # newJointAngle2 = math.acos((joint1Len**2 + joint2Len**2 - newZ**2)/(2*joint1Len*joint2Len))
    # oldJointAngle2 = math.acos((joint1Len**2 + joint2Len**2 - z**2)/(2*joint1Len*joint2Len))
    #
    # newJointAngle1 = joint2Len * math.sin(newJointAngle2) / newZ
    # print(math.sin(newJointAngle2))
    # oldJointAngle1 = joint2Len * math.sin(oldJointAngle2) / z
    #
    # return [0,math.degrees(newJointAngle1-oldJointAngle1), math.degrees(oldJointAngle2 - newJointAngle2), 0, 0, 0]
    # Results

    # Bounds are optional but can help convergence
    bounds = [(0, 2 * np.pi)] * 3

    # Solve
    result = minimize(
        objective,
        initial_angles,
        bounds=bounds,
        constraints=[
            {'type': 'ineq', 'fun': constraintJoint1},
            {'type': 'ineq', 'fun': constraintJointUpper1},
            {'type': 'ineq', 'fun': constraintJoint2},
            {'type': 'ineq', 'fun': constraintJointUpper2},
            {'type': 'ineq', 'fun': constraintJoint3},
            {'type': 'ineq', 'fun': constraintJointUpper3},
        ]
    )

    if result.success:
        alpha_opt, beta_opt, gamma_opt = result.x
        print("Solution found:")
        print(f"alpha = {np.degrees(alpha_opt):.2f}°")
        print(f"beta = {np.degrees(beta_opt):.2f}°")
        print(f"gamma = {np.degrees(gamma_opt):.2f}°")
        print(f"alpha + beta + gamma = {np.degrees(alpha_opt + beta_opt + gamma_opt):.2f}°")
        
        print(f"Desired z: {z}")
        # print(f"Initial z: {initial_gripper_coords_from_base[2]}")
        print(f"Result z: {z_from_arm([alpha_opt, beta_opt, gamma_opt])}")
        
        print(f"Desired x: {x}")
        # print(f"X from arm: {x_from_arm([alpha_opt, beta_opt, gamma_opt])}")
        # print(f"Initial arm x: {initial_arm_x}")
        # print(f"Initial gripper x from base: {initial_gripper_coords_from_base[0]}")
        print(f"Result x: {initial_gripper_coords_from_base[0] - (x_from_arm([alpha_opt, beta_opt, gamma_opt]) - initial_arm_x)}")
        
        print(f"Penalty: {(((alpha_opt + beta_opt + gamma_opt) - 270) ** 2)* 1e-8}")
    else:
        print("Optimization failed:", result.message)

    return [alpha_opt, beta_opt, gamma_opt]

def get_move_angles(camera_coords, target_coords, current_angles):
    gripper_coords = conv_camera_coords_to_gripper_coords(camera_coords, current_angles)
    degrees = move_to_position(gripper_coords, current_angles, target_coords)
    return degrees

