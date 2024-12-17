from flask import Flask, request, render_template, send_file
from flask_cors import CORS
import numpy as np
import cv2
import os

app = Flask(__name__)
CORS(app)  # Поддержка CORS

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Линейное контрастирование
def linear_contrast_manual(image):
    min_val = image.min()
    max_val = image.max()
    scaled_image = (image - min_val) / (max_val - min_val) * 255
    return scaled_image.astype(np.uint8)

# Выравнивание гистограммы
def equalize_hist_manual(image_channel):
    hist = np.bincount(image_channel.ravel(), minlength=256)
    total_pixels = image_channel.size
    cdf = np.cumsum(hist) / total_pixels
    equalized = (cdf[image_channel] * 255).astype(np.uint8)
    return equalized

# Выравнивание гистограммы (HSV)
def histogram_equalization_hsv_manual(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = equalize_hist_manual(hsv[:, :, 2])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Выравнивание гистограммы (RGB)
def histogram_equalization_rgb_manual(image):
    channels = cv2.split(image)
    eq_channels = [equalize_hist_manual(channel) for channel in channels]
    return np.stack(eq_channels, axis=-1)

# Реализация поэлементных операций
def elementwise_operation(image, operation="invert", value=50):
    """
    Выполняет поэлементные операции над изображением.
    
    :param image: входное изображение (NumPy массив).
    :param operation: тип операции (invert, add, subtract, multiply, divide).
    :param value: значение для операции (используется в add, subtract, multiply, divide).
    :return: обработанное изображение.
    """
    if operation == "invert":
        result = 255 - image
    elif operation == "add":
        result = np.clip(image + value, 0, 255)  # Добавление с ограничением по диапазону [0, 255]
    elif operation == "subtract":
        result = np.clip(image - value, 0, 255)  # Вычитание с ограничением
    elif operation == "multiply":
        result = np.clip(image * value, 0, 255)  # Умножение с ограничением
    elif operation == "divide":
        result = np.clip(image / (value if value != 0 else 1), 0, 255)  # Деление с защитой от деления на 0
    else:
        raise ValueError("Invalid operation type. Supported: invert, add, subtract, multiply, divide")
    
    return result.astype(np.uint8)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Загрузка изображения
    image = cv2.imread(filepath)
    if image is None:
        return "Failed to load image. Ensure the file is a valid image format.", 400

    # Выбор метода обработки
    method = request.form.get('method')
    if method == 'linear_contrast':
        result_image = linear_contrast_manual(image)
    elif method == 'histogram_hsv':
        result_image = histogram_equalization_hsv_manual(image)
    elif method == 'histogram_rgb':
        result_image = histogram_equalization_rgb_manual(image)
    elif method == 'elementwise':
        # Получение параметров для поэлементных операций
        operation = request.form.get('operation', 'invert')
        value = int(request.form.get('value', 50))
        result_image = elementwise_operation(image, operation=operation, value=value)
    else:
        return "Invalid method selected", 400

    # Сохранение результата
    result_path = os.path.join(RESULT_FOLDER, f"result_{file.filename}")
    cv2.imwrite(result_path, result_image)

    return send_file(result_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)