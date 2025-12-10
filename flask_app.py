import cv2
from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
import io

from src.camera_utils import decode_image, get_camera_pos_from_board
from src.movement import get_move_angles, get_initial_angles

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LATEST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "latest.jpg")
angles = get_initial_angles()
print("Initial angles:")
print(angles)


@app.route('/get_position', methods=['POST'])
def receive_image():
    global angles
    
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    img = decode_image(file_bytes)
    img, camera_position = get_camera_pos_from_board(img)
    
    print("start file writing")
    cv2.imwrite(LATEST_IMAGE_PATH, img)
    print("file writing done")
    
    angles = get_move_angles(camera_position, [0.2, camera_position[1], 0.05], angles)
    return jsonify({"message": "OK", "camera_position": camera_position.tolist()}), 200

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
    global angles
    
    print("Sending angles:", angles)
    return jsonify(angles), 200

if __name__ == '__main__':
    app.run()