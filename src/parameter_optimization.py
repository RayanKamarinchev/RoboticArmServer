import os
import cv2
import numpy as np
from scipy.optimize import minimize

from camera_utils import get_camera_position, get_marker_positions
import matplotlib.pyplot as plt

MARKER_SIZE = 0.036
MARKER_SPACING = 0.005
baseElevation = 0.132

c=0.13
e = 0.07
camera_offset_len = 0.016
delta = np.radians(75)

ANGLE_SCALE = 100.0
LENGTH_SCALE = 0.1



def rotate_vec(vec, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R @ vec


def get_rotation_matrix(alpha):
    co, si = np.cos(alpha), np.sin(alpha)
    return np.array([
        [-co, si, 0],
        [ si, co, 0],
        [0, 0, 1]
    ])


def get_translation(p1, p2, alpha):
    R = get_rotation_matrix(alpha)
    return p2 - R @ p1


def transform_arm_to_space_coords(p1, alpha, t):
    R = get_rotation_matrix(alpha)
    return R @ p1 + t


def get_arm_vectors(alpha, beta, gamma, psi, a, b):
    l2_angle = alpha + beta - np.pi

    l1 = a * np.array([np.cos(alpha), 0, np.sin(alpha)])
    l2 = b * np.array([np.cos(l2_angle), 0, np.sin(l2_angle)])

    lb = l1 + l2

    phi = np.arctan2(lb[2], lb[0])
    angle_sum = alpha + beta + gamma
    epsilon = np.pi / 2 + angle_sum - phi

    lh_x = c * np.array([np.cos(angle_sum), 0, np.sin(angle_sum)])
    lh_y = c * np.array([
        np.sin(epsilon) * np.cos(phi),
        np.cos(epsilon),
        np.sin(epsilon) * np.sin(phi)
    ])

    lh = np.cos(psi) * lh_x + np.sin(psi) * lh_y
    return lb, lh


def get_gripper_coords_and_cam_rotation_from_arm(angles, a, b):
    alpha, beta, gamma, theta, psi = angles
    psi = 0

    lb, lh = get_arm_vectors(alpha, beta, gamma, psi, a, b)
    arm_vector = lb + lh

    position = rotate_vec(arm_vector, theta)
    position[2] += baseElevation

    return position, None


def conv_camera_coords_to_gripper_coords(camera_coords, angles, coordinate_systems_angle, a, b):
    _, arm_head = get_arm_vectors(angles[0], angles[1], angles[2], angles[4], a, b)
    _, cam_vec = get_arm_vectors(angles[0], angles[1], angles[2] + delta, angles[4], a, b)

    cam_vec = cam_vec * e / c
    cam_offset = np.cross(arm_head, cam_vec)
    cam_offset = cam_offset / np.linalg.norm(cam_offset) * camera_offset_len

    disp_arm = -cam_offset - cam_vec + arm_head

    co, si = np.cos(coordinate_systems_angle), np.sin(coordinate_systems_angle)
    R = np.array([[-co, si, 0], [si, co, 0], [0, 0, 1]])

    return camera_coords + R @ disp_arm


def get_angles(vars, servo_angles):
    return [
        np.radians(servo_angles[0] + vars[0]),
        np.radians(vars[1] - servo_angles[1]),
        np.radians(servo_angles[2] + vars[2]),
        np.radians(servo_angles[3] - 30),
        0
    ]



def unpack_vars(z):
    return np.array([
        z[0] * ANGLE_SCALE,
        z[1] * ANGLE_SCALE,
        z[2] * ANGLE_SCALE,
        z[3] * LENGTH_SCALE,
        z[4] * LENGTH_SCALE,
    ])
    
def pack_z(vars):
    return np.array([
        vars[0] / ANGLE_SCALE,
        vars[1] / ANGLE_SCALE,
        vars[2] / ANGLE_SCALE,
        vars[3] / LENGTH_SCALE,
        vars[4] / LENGTH_SCALE,
    ])


def objective_scaled(z, initial_camera_position, initial_coordinate_systems_angle, imgs_info):
    vars = unpack_vars(z)

    init_angles = get_angles(vars, [100, 100, 6, 30])

    init_gripper_space = conv_camera_coords_to_gripper_coords(
        initial_camera_position,
        init_angles,
        initial_coordinate_systems_angle,
        vars[3], vars[4],
    )

    init_gripper_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(
        init_angles, vars[3], vars[4],
    )

    arm_angle = np.arctan2(init_gripper_arm[1], init_gripper_arm[0])
    corrected_angle = initial_coordinate_systems_angle - arm_angle

    translation = get_translation(
        init_gripper_arm,
        init_gripper_space,
        corrected_angle,
    )

    error = 0.0

    for servo_angles, cam_position, coord_angle in imgs_info:
        angles = get_angles(vars, servo_angles)

        real_space = conv_camera_coords_to_gripper_coords(
            cam_position,
            angles,
            coord_angle,
            vars[3], vars[4]
        )

        arm_pos, _ = get_gripper_coords_and_cam_rotation_from_arm(
            angles, vars[3], vars[4]
        )

        pred_space = transform_arm_to_space_coords(
            arm_pos,
            coord_angle,
            translation,
        )

        diff = pred_space - real_space
        diff*=100
        error += diff @ diff

    return error / len(imgs_info)



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(BASE_DIR, "views_from_servo_angles")

images = [os.path.join(folder, f) for f in os.listdir(folder)]

initial_img_path = next(x for x in images if "30_100_100_6" in x)
initial_img = cv2.imread(initial_img_path)

_, init_cam_pos, init_coord_angle, _, _, _ = get_camera_position(
    initial_img,
    get_marker_positions(MARKER_SIZE, MARKER_SPACING),
    MARKER_SIZE
)

imgs_info = []

for img_path in images:
    img = cv2.imread(img_path)
    name = os.path.splitext(os.path.basename(img_path))[0]
    angles = [int(x) for x in name.split("_")]
    servo_angles = angles[1:] + angles[:1]
    print("Img: ", img_path)
    _, cam_pos, coord_angle, _, _, _ = get_camera_position(
        img,
        get_marker_positions(MARKER_SIZE, MARKER_SPACING),
        MARKER_SIZE
    )

    imgs_info.append((servo_angles, cam_pos, coord_angle))



init_vars = np.array([-8, 211, 73, 0.119, 0.122])

z0 = np.array([
    init_vars[0] / ANGLE_SCALE,
    init_vars[1] / ANGLE_SCALE,
    init_vars[2] / ANGLE_SCALE,
    init_vars[3] / LENGTH_SCALE,
    init_vars[4] / LENGTH_SCALE
])

bounds = [
    (-2, 2),
    (1.7, 2.5),
    (0.5, 1),
    (1, 1.5),
    (1, 1.5)
]

result = minimize(
    objective_scaled,
    z0,
    args=(init_cam_pos, init_coord_angle, imgs_info),
    method="Powell",
    bounds=bounds,
    options={"xtol": 1e-6, "ftol": 1e-6},
)

optimized_vars = unpack_vars(result.x)

print("Success:", result.success)
print("Final error:", result.fun)
initial_error = objective_scaled(
    z0,
    init_cam_pos,
    init_coord_angle,
    imgs_info
)
print("Initial error:", initial_error)
print("Optimized vars:", optimized_vars)
optim_err = objective_scaled(
    result.x,
    init_cam_pos,
    init_coord_angle,
    imgs_info
)
print("Optimized error:", optim_err)

def collect_points(vars, imgs_info, init_cam_pos, init_coord_angle, prev_errors = None):
    real_pts = []
    pred_pts = []
    erros = []

    init_angles = get_angles(vars, [100, 100, 6, 30])

    init_gripper_space = conv_camera_coords_to_gripper_coords(
        init_cam_pos, init_angles, init_coord_angle,
        vars[3], vars[4]
    )

    init_gripper_arm, _ = get_gripper_coords_and_cam_rotation_from_arm(
        init_angles, vars[3], vars[4]
    )

    arm_angle = np.arctan2(init_gripper_arm[1], init_gripper_arm[0])
    corrected_angle = init_coord_angle - arm_angle

    translation = get_translation(
        init_gripper_arm,
        init_gripper_space,
        corrected_angle
    )

    i = 0
    for servo_angles, cam_pos, coord_angle in imgs_info:
        angles = get_angles(vars, servo_angles)

        real_space = conv_camera_coords_to_gripper_coords(
            cam_pos, angles, coord_angle,
            vars[3], vars[4]
        )

        arm_pos, _ = get_gripper_coords_and_cam_rotation_from_arm(
            angles, vars[3], vars[4]
        )

        pred_space = transform_arm_to_space_coords(
            arm_pos, coord_angle, translation
        )
        
        diff = real_space-pred_space
        err = diff @ diff
        erros.append(err)
        
        # if(servo_angles == [100, 100, 20, 15]):
        #     print("Cam: ", cam_pos)
        #     print("Gripper space: ", real_space)
        #     print("pred space: ", pred_space)
        #     print("coord angle: ", coord_angle)
        #     print("coord angle: ", np.degrees(coord_angle))
        #     real_pts.append(real_space)
        #     pred_pts.append(pred_space)
        
        # if (err > 0.0002):
        #     print("Img servos: ", servo_angles)
        #     print("err: ", err)
        #     erros.append(err)
        
        
        # if(prev_errors is None):
        #     real_pts.append(real_space)
        #     pred_pts.append(pred_space)
        # else:
        #     if (prev_errors[i] > err):
        #         real_pts.append(real_space)
        #         pred_pts.append(pred_space)
                
        real_pts.append(real_space)
        pred_pts.append(pred_space)
        i+=1
    

    return np.array(real_pts), np.array(pred_pts), erros

_, pred_init, errors_init = collect_points(
    init_vars, imgs_info, init_cam_pos, init_coord_angle
)
print("------------------------------")
# Final prediction
real_pts, pred_final, errors_final = collect_points(
    optimized_vars, imgs_info, init_cam_pos, init_coord_angle, errors_init
)

print([float(round(x, 7)) for x in errors_init])
print([float(round(x, 7)) for x in errors_final])

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# ax.set_xlim(0, 0.3)
# ax.set_ylim(0, 0.1)
# ax.set_zlim(0, 0.4)

ax.scatter(
    real_pts[:, 0], real_pts[:, 1], real_pts[:, 2],
    label="Real", marker="o"
)

# ax.scatter(
#     pred_init[:, 0], pred_init[:, 1], pred_init[:, 2],
#     label="Initial prediction", marker="^"
# )

ax.scatter(
    pred_final[:, 0], pred_final[:, 1], pred_final[:, 2],
    label="Final prediction", marker="x"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.tight_layout()
plt.show()







#New method
initial_pred = np.array([-8, 211, 73, 0.119, 0.122])
prev_pred = [-3.20496343,  205.360898,  68.3649316,  0.120533090, 0.127455405]
# alpha_range = np.arange(-15, 1)
alpha_range = np.arange(-10, 1)
beta_range = np.arange(200, 221)
gamma_range = np.arange(60, 81)
a_range = np.arange(0.114, 0.125, 0.002)
b_range = np.arange(0.115, 0.13, 0.002)
print(alpha_range)
print(beta_range)
print(gamma_range)
print(a_range)
print(b_range)
min_loss = np.inf
best_vars = [-3, 205, 69, 0.12, 0.127]

# for i, alpha in enumerate(alpha_range):
#     for j, beta in enumerate(beta_range):
#         for gamma in gamma_range:
#             for a in a_range:
#                 for b in b_range:
#                     loss = objective_scaled(pack_z([alpha, beta, gamma, a, b]), init_cam_pos, init_coord_angle, imgs_info)
#                     if loss < min_loss:
#                         min_loss = loss
#                         best_vars = [alpha, beta, gamma, a, b]   
#         print(((j/len(beta_range) + i)/len(alpha_range))*100) 
#     print("alpha")          
    
print("Initial error:", initial_error)
print("Optimized error:", optim_err)
print("New method error:", min_loss)
print("best vars: ", best_vars)