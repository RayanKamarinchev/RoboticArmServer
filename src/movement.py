import math
import numpy as np
from scipy.optimize import minimize

a = 11.7
b = 12.2
c = 13
baseElevation = 13

def move(x, y, z):
    def objective(vars):
        alpha, beta, gamma = vars
        # Equation residuals
        eq1 = a * np.sin(alpha) - b * np.sin(alpha + beta) + c * np.sin(alpha + beta + gamma) - z + baseElevation
        eq2 = a * np.cos(alpha) - b * np.cos(alpha + beta) + c * np.cos(alpha + beta + gamma) - x

        total_angle_deg = np.degrees(x + y + z)
        penalty = (total_angle_deg - 270) ** 2

        return eq1 ** 2 + eq2 ** 2 + 0.01 * penalty

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
        return 215 - np.degrees(alpha)

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
    # Initial guess (in radians)
    x0 = np.radians([88, 178, 74])

    # Bounds are optional but can help convergence
    bounds = [(0, 2 * np.pi)] * 3

    # Solve
    result = minimize(
        objective,
        x0,
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
        print(f"x = {np.degrees(alpha_opt):.2f}°")
        print(f"y = {np.degrees(beta_opt):.2f}°")
        print(f"z = {np.degrees(gamma_opt):.2f}°")
        print(f"x + y + z = {np.degrees(alpha_opt + beta_opt + gamma_opt):.2f}°")
        print(f"z: {a * np.sin(alpha_opt) - b * np.sin(alpha_opt + beta_opt) + c * np.sin(alpha_opt + beta_opt + gamma_opt) + baseElevation}")
        print(f"x: {a * np.cos(alpha_opt) - b * np.cos(alpha_opt + beta_opt) + c * np.cos(alpha_opt + beta_opt + gamma_opt)}")
    else:
        print("Optimization failed:", result.message)

    return [np.degrees(alpha_opt), np.degrees(beta_opt), np.degrees(gamma_opt)]


#13.4-13.6
# x, y, z = (13.5, 0, 32.4)
# move(x,y,z)
#
# alpha,beta,gamma = np.radians([88, 178, 74])
# print(a * np.sin(alpha) - b * np.sin(alpha + beta) + c * np.sin(alpha + beta + gamma) + 13)
# print(a * np.cos(alpha) - b * np.cos(alpha + beta) + c * np.cos(alpha + beta + gamma))
