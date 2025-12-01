import cv2
import numpy as np
import imutils
import sys


def order_corner_points(corners):
  corners = [(corner[0][0], corner[0][1]) for corner in corners]
  top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
  return (top_l, top_r, bottom_r, bottom_l)

def perspective_transform(image, corners):
  top_l, top_r, bottom_r, bottom_l = corners

  width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
  width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
  width = max(int(width_A), int(width_B))

  height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
  height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
  height = max(int(height_A), int(height_B))

  dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                  [0, height - 1]], dtype = "float32")

  ordered_corners = np.array(corners, dtype="float32")

  matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

  return cv2.warpPerspective(image, matrix, (width, height))


def get_camera_position(file):
    file_bytes = np.frombuffer(file.read(), np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (9, 9), 0)

    thresh = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    thresh = cv2.bitwise_not(thresh)

    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse = True)

    peri = cv2.arcLength(cnts[0], True)
    biggest_cnt = cv2.approxPolyDP(cnts[0], 0.015 * peri, True)

    imagedrawed = image.copy()
    imagedrawed = cv2.drawContours(imagedrawed, [biggest_cnt], -1, (255, 0, 0), 5)
    cv2.imwrite('transformed.jpg', imagedrawed)
    scaled_image = cv2.resize(imagedrawed, None, fx=0.5, fy=0.5)
    cv2.imshow('efwfew', scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(biggest_cnt) < 4:
        return [-1, -1, -1]
    main_corners = order_corner_points(biggest_cnt)

    transformed = perspective_transform(image.copy(), main_corners)


    SQUARE_SIZE = 3*8

    objp = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [1, 1, 0],
                     [0, 1, 0]], dtype=np.float32)
    objp *= SQUARE_SIZE
    main_corners = np.array(main_corners).astype(np.float32).reshape(-1, 1, 2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, main_corners, (11, 11), (-1, -1), criteria)

    h, w = gray.shape
    camera_matrix = np.array([[1566, 0, w / 2],
                          [0, 1575, h / 2],
                          [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1))  # assume no distortion for simplicity

    ret, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, dist_coeffs)

    world_point_pose = np.array([0, 0, 0])

    world_point_pose_mm = world_point_pose * 1000

    def matrix_from_vecs(tvec, rvec):
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    board_pose_in_camera_frame = matrix_from_vecs(tvec, rvec)

    pose_of_camera_in_board_frame = np.linalg.inv(board_pose_in_camera_frame)

    transform_from_board_to_world = np.eye(4)
    transform_from_board_to_world[:3, 3] = world_point_pose_mm

    camera_in_world = np.dot(transform_from_board_to_world, pose_of_camera_in_board_frame)
    return camera_in_world[:3, 3]