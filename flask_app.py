from flask import Flask, request, jsonify
import os
from datetime import datetime
import io

from src.camera_get_position import get_camera_position
from src.movement import move

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/get_position', methods=['POST'])
def receive_image():
    if 'imageFile' not in request.files:
        print("FILES:", request.files)
        return jsonify({"error": "No file part"}), 400

    file = request.files['imageFile']
    file_bytes = file.read()
    print("Received:", len(file_bytes), "bytes")

    filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(save_path, 'wb') as f:
        f.write(file_bytes)

    return jsonify({"message": "OK", "bytes": len(file_bytes)})

@app.route('/get_movements', methods=['GET'])
def receive_data():
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    z = float(request.args.get('z'))
    moves = move(x, y, z)

    return jsonify(moves), 200


if __name__ == '__main__':
    app.run()