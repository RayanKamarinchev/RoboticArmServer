import cv2
from flask import Flask, request, jsonify, send_file, render_template
import os
from datetime import datetime
import numpy as np
import json

from src.box_detection import get_box_coordinates
from src.camera_utils import decode_image, get_camera_position, get_marker_positions, get_camera_matrix_and_dist_coeffs
from src.movement import get_move_angles, get_initial_angles, conv_camera_coords_to_gripper_coords
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #TODO

app = Flask(__name__)

MARKER_SIZE=0.036
MARKER_SPACING=0.005
BASELINE=0.02

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
INSTRUCTIONS_DIR = os.path.join(UPLOAD_FOLDER, "instructions.json")
LATEST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "latest.jpg")
IMAGE_1_PATH = os.path.join(UPLOAD_FOLDER, "image1.jpg")
IMAGE_2_PATH = os.path.join(UPLOAD_FOLDER, "image2.jpg")
IMAGE_PATHS = [IMAGE_1_PATH, IMAGE_2_PATH]
DEBUG_DATA_PATH = "src/data.csv"
instructions = []
flag = False
counter = 0

def prepare_instructions(img):
    global flag
    global instructions
    
    copy_img, camera_position, coordinate_systems_angle, R, rvec, tvec = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    print("Camera position:", camera_position) 
    
    target_position = conv_camera_coords_to_gripper_coords(camera_position, get_initial_angles(), coordinate_systems_angle)
    
    angles = get_move_angles(camera_position, target_position, get_initial_angles(), coordinate_systems_angle)
    first_angles = angles.copy()
    camera_matrix, dist_coeffs = get_camera_matrix_and_dist_coeffs()
    
    with open(INSTRUCTIONS_DIR, "r") as f:
        instructions_data = json.load(f)
    
    # instructions = []
    
    # instructions.append(["move", *angles])
    # instructions.append(["wait", 1])

    # target_position = get_box_coordinates(img, camera_position, R, camera_matrix, dist_coeffs, rvec, tvec)[0]
    # angles = get_move_angles(camera_position, target_position, get_initial_angles(), coordinate_systems_angle)
    
    # instructions.append(["move", *angles])
    # instructions.append(["wait", 1])
    # instructions.append(["grip", 1])
    # instructions.append(["wait", 1])
    # instructions.append(["move", *first_angles])
    # instructions.append(["wait", 30])
    # instructions.append(["grip", 0])
    
    # for line in instructions_data:
    #     if line[0] == "move":
    #         angles = get_move_angles(camera_position, line[1], get_initial_angles(), coordinate_systems_angle)
    #         instructions.append(["move", *angles])
    #     else:
    #         instructions.append(line)
    
    flag = True 
    
    return camera_position

@app.route('/get_position', methods=['POST'])
def receive_image():
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    cv2.imwrite(LATEST_IMAGE_PATH, img)
    camera_position = prepare_instructions(img)
    
    return jsonify({"message": "OK", "camera_position": camera_position.tolist()}), 200

@app.route('/debug', methods=['POST'])
def debug():
    global counter
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    angles_str = request.form.get("angles")
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    
    img, camera_position, coordinate_systems_angle, _ = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    print("Camera position:", camera_position.tolist()) 
    img_name = f"{counter}_image_{angles_str}.jpg"
    counter+=1
    image_save_dir = os.path.join(UPLOAD_FOLDER, img_name)
    cv2.imwrite(image_save_dir, img)

    with open(DEBUG_DATA_PATH, "a") as f:
        f.write(f"{camera_position.tolist()},{image_save_dir},{angles_str},{coordinate_systems_angle}\n")
    
    return jsonify({"message": "Debug endpoint reached"}), 200

@app.route("/get_depth", methods=["POST"])
def get_depth():
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    angles = request.form.get("angles")
    image_num = int(file.filename.split('_')[-1].split('.')[0])
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    cv2.imwrite(IMAGE_PATHS[image_num-1], img)
    img, camera_position, coordinate_systems_angle, _ = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    print("Camera position:", camera_position)
    
    if (image_num == 2):
        prepare_instructions(img)
        
    return jsonify({"message": "OK", "camera_position": camera_position}), 200
    
    

@app.route('/latest.jpg')
def latest_image():
    if os.path.exists(LATEST_IMAGE_PATH):
        return send_file(LATEST_IMAGE_PATH, mimetype='image/jpeg')
    return "No image received yet.", 404

@app.route('/get_movements', methods=['GET'])
def receive_data():
    global instructions
    global flag
    
    if not flag:
        return jsonify({"error": "No angles calculated yet."}), 400
    
    flag = False
    print("Sending instructions:", instructions)
    return jsonify(instructions), 200

#User Interface:

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)