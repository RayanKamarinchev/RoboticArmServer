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
    # if 'imageFile' not in request.files:
    #     return jsonify({'error': 'No file part'}), 400

    # file = request.files['imageFile']

    # file_bytes = request.files['imageFile'].read()
    file_bytes = request.data

    # file_stream = io.BytesIO(file_bytes)

    # pos = get_camera_position(file_stream)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"photo_{timestamp}_.jpg"
    save_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(save_path, 'wb') as f:
        f.write(file_bytes)

    return jsonify({"message": "Image received and saved successfully."})

@app.route('/get_movements', methods=['GET'])
def receive_data():
    x = float(request.args.get('x'))
    y = float(request.args.get('y'))
    z = float(request.args.get('z'))
    moves = move(x, y, z)

    return jsonify(moves), 200


if __name__ == '__main__':
    app.run()