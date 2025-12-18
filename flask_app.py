import cv2
from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
import io
import numpy as np
import json

from src.camera_utils import decode_image, get_camera_position, get_marker_positions
from src.movement import get_move_angles, get_initial_angles, conv_camera_coords_to_gripper_coords

app = Flask(__name__)

MARKER_SIZE=0.036
MARKER_SPACING=0.005

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
INSTRUCTIONS_DIR = os.path.join(UPLOAD_FOLDER, "instructions.json")
LATEST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "latest.jpg")
instructions = []
flag = False


@app.route('/get_position', methods=['POST'])
def receive_image():
    global flag
    global instructions
    
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    cv2.imwrite(LATEST_IMAGE_PATH, img)
    img, camera_position = get_camera_position(img, get_marker_positions(MARKER_SIZE, MARKER_SPACING), MARKER_SIZE)
    print("Camera position:", camera_position)
    
    target_position = conv_camera_coords_to_gripper_coords(camera_position, get_initial_angles())
    
    angles = get_move_angles(camera_position, target_position, get_initial_angles())
    with open(INSTRUCTIONS_DIR, "r") as f:
        instructions_data = json.load(f)
    
    instructions = []
    
    instructions.append(["move", *angles])
    instructions.append(["wait", 1])
    
    for line in instructions_data:
        if line[0] is "move":
            angles = get_move_angles(camera_position, line[1], get_initial_angles())
            instructions.append(["move", *angles])
        else:
            instructions.append(line)
    
    flag = True
    
    return jsonify({"message": "OK", "camera_position": camera_position}), 200

@app.route('/latest.jpg')
def latest_image():
    if os.path.exists(LATEST_IMAGE_PATH):
        return send_file(LATEST_IMAGE_PATH, mimetype='image/jpeg')
    return "No image received yet.", 404

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>ESP32-CAM Live View</title>
            <meta http-equiv="refresh" content="1">
        </head>
        <body>
            <h1>Latest ESP32-CAM Image</h1>
            <img src="latest.jpg" width="640">
        </body>
    </html>
    """

@app.route('/get_movements', methods=['GET'])
def receive_data():
    global instructions
    global flag
    
    if not flag:
        return jsonify({"error": "No angles calculated yet."}), 400
    
    flag = False
    print("Sending instructions:", instructions)
    return jsonify(instructions), 200

if __name__ == '__main__':
    app.run()