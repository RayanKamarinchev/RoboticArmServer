from flask import Flask, request, jsonify, send_file
import os
from datetime import datetime
import io

from src.camera_get_position import get_camera_position
from src.movement import move

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
LATEST_IMAGE_PATH = os.path.join(UPLOAD_FOLDER, "latest.jpg")


@app.route('/get_position', methods=['POST'])
def receive_image():
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    print(BASE_DIR)
    print(UPLOAD_FOLDER)
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(save_path, 'wb') as f:
        f.write(file_bytes)

    with open(LATEST_IMAGE_PATH, 'wb') as f:
        f.write(file_bytes)

    return jsonify({"message": "OK", "bytes": len(file_bytes)})

@app.route('/latest')
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
            <img src="/uploads/latest.jpg" width="640">
        </body>
    </html>
    """

@app.route('/get_movements', methods=['GET'])
def receive_data():
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    z = float(request.args.get('z'))
    moves = move(x, y, z)

    return jsonify(moves), 200


if __name__ == '__main__':
    app.run()