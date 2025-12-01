from camera_get_position import get_camera_position
from werkzeug.datastructures import FileStorage

img = 'chessboard_unordered.jpg'
with open('imgs/' + img, 'rb') as f:
    file_storage = FileStorage(stream=f, filename=img, content_type='image/jpeg')
    print(get_camera_position(file_storage))